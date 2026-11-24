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
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class GuidedLowConfidenceSeedPromotionManifestV1Tests
    {
        string tempRoot;

        static string FixedProfileDirectory => Path.Combine(
            Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles",
            "step9l_current_user_multi_prototype_v1_0");
        static string GlobalTemplateDirectory => Path.Combine(
            Application.streamingAssetsPath, "RecognitionCoreV1", "Templates");
        static string GlobalThresholdPath => Path.Combine(
            Application.streamingAssetsPath, "RecognitionCoreV1", "RuntimeGate",
            "frozen_threshold_config.json");

        [SetUp]
        public void SetUp()
        {
            tempRoot = Path.Combine(Path.GetTempPath(), "wis_step10k_v2_3_" + Guid.NewGuid().ToString("N"));
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(tempRoot)) Directory.Delete(tempRoot, true);
        }

        [Test]
        public void RawReviewOnlyThreeOrMoreRemainUntrustedUntilExplicitCommit()
        {
            var repository = NewRepository();
            SaveReview(repository, "LOVE", "review-1", 1, 0.30, -0.10);
            SaveReview(repository, "LOVE", "review-2", 2, 0.20, -0.20);
            SaveReview(repository, "LOVE", "review-3", 3, 0.10, -0.30);
            SaveReview(repository, "LOVE", "review-4", 4, 0.05, -0.40);
            Assert.AreEqual(0, repository.ReloadTrustedAndVerify().CountForWord("LOVE"));
            Assert.IsFalse(File.Exists(ManifestPath(repository, "LOVE")));
        }

        [Test]
        public void TwoCandidatesFailWithoutCreatingManifestTarget()
        {
            var repository = NewRepository();
            SaveReview(repository, "LOVE", "two-1", 1);
            SaveReview(repository, "LOVE", "two-2", 2);
            GuidedEnrollmentSeedCommitResult result = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.InsufficientCandidates, result.Status);
            Assert.IsFalse(File.Exists(ManifestPath(repository, "LOVE")));
            Assert.AreEqual(0, repository.TrustedInventory.CountForWord("LOVE"));
        }

        [Test]
        public void ExplicitCommitSelectsDeterministicExactThreeByApprovedOrdering()
        {
            var repository = NewRepository();
            SaveReview(repository, "LOVE", "rank-fourth", 1, 0.10, -0.10);
            SaveReview(repository, "LOVE", "rank-first", 2, 0.50, -0.50);
            SaveReview(repository, "LOVE", "rank-third", 3, 0.30, -0.05);
            SaveReview(repository, "LOVE", "rank-second", 4, 0.40, -0.55);
            GuidedEnrollmentSeedCommitResult result = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.Committed, result.Status);
            CollectionAssert.AreEqual(new[] { "rank-first", "rank-second", "rank-third" },
                result.SelectedCaptureSessionIds);
            Assert.AreEqual(3, CountOccurrences(File.ReadAllText(result.ManifestPath), "selection_order"));
        }

        [Test]
        public void SelectionTieUsesCaptureSessionIdOrdinalAscending()
        {
            var repository = NewRepository();
            foreach (string id in new[] { "tie-c", "tie-a", "tie-d", "tie-b" })
                SaveReview(repository, "LOVE", id, id[4], 0.25, -0.10);
            GuidedEnrollmentSeedCommitResult result = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            CollectionAssert.AreEqual(new[] { "tie-a", "tie-b", "tie-c" }, result.SelectedCaptureSessionIds);
        }

        [Test]
        public void OnlyManifestSelectedReviewCandidatesReceiveGuidedTrustBasis()
        {
            var repository = NewRepository();
            for (int sample = 1; sample <= 5; sample++)
                SaveReview(repository, "LOVE", "selected-basis-" + sample, sample,
                    1.0 - sample * 0.1, -sample * 0.05);
            repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();
            Assert.AreEqual(3, inventory.CountForWord("LOVE"));
            Assert.AreEqual(3, inventory.Candidates.Count(candidate =>
                candidate.Word == "LOVE" &&
                candidate.TrustBasis == TrustedPersonalizationTrustBasis.GuidedLowConfidenceSeed));
            Assert.AreEqual(2, inventory.RawCounts.ReviewOnly["LOVE"] - inventory.CountForWord("LOVE"));
            Assert.AreEqual(2, inventory.ExcludedDuplicateOrUntrustedCount);
        }

        [Test]
        public void StrictEligibleTrustPathIsPreservedWithoutManifest()
        {
            var repository = NewRepository();
            SaveStrict(repository, "GREETING", "strict-one", 1);
            TrustedPersonalizationCandidate candidate = repository.TrustedInventory.Candidates.Single();
            Assert.AreEqual(TrustedPersonalizationTrustBasis.StrictEligible, candidate.TrustBasis);
            Assert.IsFalse(File.Exists(ManifestPath(repository, "GREETING")));
        }

        [Test]
        public void StrictQuorumMakesGuidedManifestNotRequiredAndCreatesNothing()
        {
            var repository = NewRepository();
            for (int sample = 1; sample <= 3; sample++)
                SaveStrict(repository, "GREETING", "strict-quorum-" + sample, sample);
            GuidedEnrollmentSeedCommitResult result = repository.TryCommitGuidedEnrollmentSeedQuorum("GREETING");
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.NotRequired, result.Status);
            Assert.IsFalse(File.Exists(ManifestPath(repository, "GREETING")));
            Assert.AreEqual(3, repository.TrustedInventory.CountForWord("GREETING"));
        }

        [TestCase("capture")]
        [TestCase("sequence")]
        [TestCase("source")]
        [TestCase("logical")]
        public void DuplicateIdentityFamilyCannotCreateTrustedSeedQuorum(string family)
        {
            var repository = NewRepository();
            PracticeRecognitionAttempt first = Attempt("duplicate-first", "LOVE",
                GuidedQuestOutcomeKind.LowConfidenceCorrect, 1);
            Assert.AreEqual(CandidateStorageStatus.Saved, SaveReview(repository, first).Status);
            if (family == "capture")
            {
                Assert.AreEqual(CandidateStorageStatus.Duplicate, SaveReview(repository, first).Status);
            }
            else if (family == "sequence")
            {
                PracticeRecognitionAttempt invalid = Attempt("duplicate-new-id", "LOVE",
                    GuidedQuestOutcomeKind.LowConfidenceCorrect, 2,
                    sequenceSampleId: first.CaptureSessionId);
                Assert.AreEqual(CandidateStorageStatus.Failed, SaveReview(repository, invalid).Status);
            }
            else
            {
                string source = first.NormalizedSequence.SourceNpzSha256;
                string logical = first.NormalizedSequence.LogicalPayloadSha256;
                for (int sample = 2; sample <= 3; sample++)
                    SaveReview(repository, Attempt("duplicate-" + sample, "LOVE",
                        GuidedQuestOutcomeKind.LowConfidenceCorrect, sample,
                        sourceHash: family == "source" ? source : null,
                        logicalHash: family == "logical" ? logical : null));
                Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.DuplicateIdentity,
                    repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE").Status);
            }
            Assert.IsFalse(File.Exists(ManifestPath(repository, "LOVE")));
        }

        [Test]
        public void MalformedManifestFailsClosedOnStrictReload()
        {
            var repository = RepositoryWithReviewQuorum("LOVE");
            Directory.CreateDirectory(Path.GetDirectoryName(ManifestPath(repository, "LOVE")));
            File.WriteAllText(ManifestPath(repository, "LOVE"), "{ malformed");
            Assert.Throws<ArgumentException>(() => repository.ReloadTrustedAndVerify());
        }

        [TestCase("metadata")]
        [TestCase("sequence")]
        [TestCase("checksum")]
        public void CandidateMetadataSequenceOrChecksumTamperFailsClosed(string part)
        {
            var repository = RepositoryWithReviewQuorum("LOVE");
            string candidate = CandidateDirectories(repository).First();
            string file = part == "metadata" ? "candidate_metadata.json" :
                part == "sequence" ? "normalized_sequence.wisrq" : "checksum_manifest.json";
            File.AppendAllText(Path.Combine(candidate, file), "tamper");
            Assert.Catch<Exception>(() => repository.ReloadTrustedAndVerify());
            Assert.IsFalse(File.Exists(ManifestPath(repository, "LOVE")));
        }

        [Test]
        public void ManifestIdentityTamperFailsClosed()
        {
            var repository = RepositoryWithReviewQuorum("LOVE");
            GuidedEnrollmentSeedCommitResult committed = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            string text = File.ReadAllText(committed.ManifestPath);
            string capture = committed.SelectedCaptureSessionIds[0];
            File.WriteAllText(committed.ManifestPath, text.Replace(capture, "x" + capture.Substring(1)));
            Assert.Throws<InvalidDataException>(() => repository.ReloadTrustedAndVerify());
        }

        [Test]
        public void AtomicPreMoveFailureLeavesNoTargetAndCleansTemporaryFile()
        {
            bool injected = false;
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot, point =>
            {
                if (point == CandidateRepositoryFailurePoint.BeforeGuidedEnrollmentSeedManifestMove)
                {
                    injected = true;
                    throw new IOException("approved injected pre-move failure");
                }
            });
            for (int sample = 1; sample <= 3; sample++)
                SaveReview(repository, "LOVE", "rollback-" + sample, sample);
            GuidedEnrollmentSeedCommitResult result = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            Assert.IsTrue(injected);
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.Failed, result.Status);
            Assert.IsFalse(File.Exists(ManifestPath(repository, "LOVE")));
            string seedDirectory = Path.GetDirectoryName(ManifestPath(repository, "LOVE"));
            Assert.IsFalse(Directory.Exists(seedDirectory) && Directory.EnumerateFileSystemEntries(seedDirectory).Any());
        }

        [Test]
        public void ExistingIdenticalManifestIsIdempotentAndDoesNotWrite()
        {
            var repository = RepositoryWithReviewQuorum("LOVE");
            GuidedEnrollmentSeedCommitResult first = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            string before = Sha256(File.ReadAllBytes(first.ManifestPath));
            DateTime stamp = File.GetLastWriteTimeUtc(first.ManifestPath);
            GuidedEnrollmentSeedCommitResult second = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.AlreadyCommitted, second.Status);
            Assert.AreEqual(before, Sha256(File.ReadAllBytes(first.ManifestPath)));
            Assert.AreEqual(stamp, File.GetLastWriteTimeUtc(first.ManifestPath));
        }

        [Test]
        public void ExistingDifferentManifestFailsAndIsNeverOverwritten()
        {
            var repository = RepositoryWithReviewQuorum("LOVE");
            string path = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE").ManifestPath;
            string changed = File.ReadAllText(path).Replace("\"selection_order\": 1", "\"selection_order\": 9");
            File.WriteAllText(path, changed);
            string before = Sha256(File.ReadAllBytes(path));
            GuidedEnrollmentSeedCommitResult result = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.Failed, result.Status);
            Assert.AreEqual(before, Sha256(File.ReadAllBytes(path)));
        }

        [Test]
        public void ConstructorAndReloadNeverAutoCreateManifest()
        {
            RepositoryWithReviewQuorum("LOVE");
            var reconstructed = NewRepository();
            reconstructed.ReloadAndVerify();
            reconstructed.ReloadTrustedAndVerify();
            Assert.IsFalse(Directory.Exists(Path.GetDirectoryName(ManifestPath(reconstructed, "LOVE"))));
        }

        [Test]
        public void SessionLowConfidenceSaveExplicitCommitAndStrictReloadAreOnlyCountingPath()
        {
            var repository = NewRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            for (int sample = 1; sample <= 3; sample++)
            {
                adapter.Next = Attempt("session-lcc-" + sample, "GREETING",
                    GuidedQuestOutcomeKind.LowConfidenceCorrect, sample);
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());
                Assert.AreEqual(sample < 3
                        ? GuidedEnrollmentSeedCommitStatus.InsufficientCandidates
                        : GuidedEnrollmentSeedCommitStatus.Committed,
                    session.LastGuidedSeedCommitResult.Status);
                Assert.AreEqual(sample < 3 ? 0 : 3, session.CurrentTrustedSampleCount);
                if (sample < 3) Assert.IsTrue(session.Retry());
            }
            Assert.IsTrue(session.IsCurrentWordComplete);
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.Personalized, session.CurrentEnrollmentState);
        }

        [TestCase(CandidateStorageStatus.Failed)]
        [TestCase(CandidateStorageStatus.Duplicate)]
        public void SessionSaveFailureOrDuplicateNeverCallsPromotion(CandidateStorageStatus status)
        {
            var repository = new FakePromotionRepository { SaveStatus = status };
            var adapter = new FakeAdapter { Next = Attempt("not-saved", "GREETING",
                GuidedQuestOutcomeKind.LowConfidenceCorrect, 1) };
            var session = new PetRobotSignPracticeSession(adapter, repository);
            session.StartExpression();
            session.FinishExpression();
            Assert.AreEqual(0, repository.CommitCalls);
            Assert.AreEqual(0, session.CurrentTrustedSampleCount);
        }

        [TestCase(GuidedQuestOutcomeKind.WrongWord)]
        [TestCase(GuidedQuestOutcomeKind.TechnicalFailure)]
        public void WrongWordAndTechnicalFailureNeverCallPromotion(GuidedQuestOutcomeKind kind)
        {
            var repository = new FakePromotionRepository();
            var adapter = new FakeAdapter
            {
                Next = kind == GuidedQuestOutcomeKind.TechnicalFailure
                    ? TechnicalAttempt("technical", "GREETING")
                    : Attempt("wrong", "GREETING", kind, 1),
            };
            var session = new PetRobotSignPracticeSession(adapter, repository);
            session.StartExpression();
            session.FinishExpression();
            Assert.AreEqual(0, repository.CommitCalls);
            Assert.AreEqual(0, repository.SaveCalls);
        }

        [Test]
        public void RuntimeActiveViewRefreshCannotCreatePromotionManifest()
        {
            var repository = RepositoryWithReviewQuorum("LOVE");
            Dictionary<string, string> before = HashTree(repository.RootDirectory);
            TrustedPersonalizationActiveViewService service = NewActiveViewService(repository);
            Assert.IsTrue(service.TryRefresh().Succeeded);
            Assert.IsFalse(service.RuntimeObservationCanMutateEnrollment);
            CollectionAssert.AreEquivalent(before, HashTree(repository.RootDirectory));
            Assert.IsFalse(File.Exists(ManifestPath(repository, "LOVE")));
        }

        [Test]
        public void ProductionRepositoryRemainsEightTwentyFiveTrustedTwoTwoOneWithoutManifest()
        {
            string root = PersonalizationCandidateRepository.DefaultRootDirectory;
            Dictionary<string, string> before = HashTree(root);
            var repository = new PersonalizationCandidateRepository("local_learner");
            Assert.AreEqual(8, repository.Counts.TotalCandidateCount);
            AssertCounts(repository.TrustedInventory, 2, 2, 1);
            Assert.AreEqual(25, Directory.GetFiles(root, "*", SearchOption.AllDirectories).Length);
            Assert.IsFalse(Directory.Exists(Path.Combine(root, "local_learner", ".guided_enrollment_seeds")));
            CollectionAssert.AreEquivalent(before, HashTree(root));
        }

        [Test]
        public void CopiedPhysicalRootPreservesAllOriginalSeventyThreeCandidateFiles()
        {
            string original = PhysicalRoot();
            Dictionary<string, string> before = HashTree(original);
            CopyTree(original, tempRoot);
            CollectionAssert.AreEquivalent(before, HashTree(tempRoot));
            Assert.AreEqual(73, before.Count);
            var repository = NewRepository();
            Assert.AreEqual(24, repository.Counts.TotalCandidateCount);
            AssertCounts(repository.TrustedInventory, 3, 0, 0);
        }

        [Test]
        public void CopiedPhysicalRootCreatesOnlyLoveAndThankYouManifestsWithExpectedSelectionsAndGate()
        {
            CopyTree(PhysicalRoot(), tempRoot);
            var repository = NewRepository();
            GuidedEnrollmentSeedCommitResult greeting = repository.TryCommitGuidedEnrollmentSeedQuorum("GREETING");
            GuidedEnrollmentSeedCommitResult love = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            GuidedEnrollmentSeedCommitResult thanks = repository.TryCommitGuidedEnrollmentSeedQuorum("THANK_YOU");
            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.NotRequired, greeting.Status);
            CollectionAssert.AreEqual(new[]
            {
                "a4560bb9-6277-4718-8e38-73e793177f29",
                "b5560fc6-45b6-4eb3-a036-bb3477d02e52",
                "a7c5b5f1-346e-43d3-99c4-891bafa4978b",
            }, love.SelectedCaptureSessionIds);
            CollectionAssert.AreEqual(new[]
            {
                "7c232347-1194-455d-97da-bebffeb7de7e",
                "99a70e9e-6f9a-44c5-a7b8-fd10085cab1b",
                "37a36a81-83b2-48c0-a27b-092350a991b9",
            }, thanks.SelectedCaptureSessionIds);
            Assert.IsFalse(File.Exists(ManifestPath(repository, "GREETING")));
            Assert.AreEqual(2, Directory.GetFiles(
                Path.GetDirectoryName(ManifestPath(repository, "LOVE")), "*.json").Length);
            AssertCounts(repository.ReloadTrustedAndVerify(), 3, 3, 3);
            Assert.IsTrue(repository.TrustedInventory.OverallEnrollmentGateReady);
        }

        [Test]
        public void CopiedPhysicalRootActiveViewUsesThreeCandidatesAndSixTotalPrototypesPerWord()
        {
            CopyTree(PhysicalRoot(), tempRoot);
            var repository = NewRepository();
            repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            repository.TryCommitGuidedEnrollmentSeedQuorum("THANK_YOU");
            TrustedPersonalizationActiveViewService service = NewActiveViewService(repository);
            TrustedActiveViewBuildResult result = service.TryRefresh();
            Assert.IsTrue(result.Succeeded);
            foreach (string word in new[] { "GREETING", "LOVE", "THANK_YOU" })
            {
                Assert.AreEqual(3, result.ActiveView.CandidateCaptureIds[word].Count);
                Assert.AreEqual(6, result.ActiveView.Profile.Prototypes[word].Count);
            }
            Assert.AreEqual(0, repository.TrustedInventory.Candidates.Count(candidate =>
                candidate.TrustBasis == TrustedPersonalizationTrustBasis.GuidedLowConfidenceSeed &&
                !result.ActiveView.CandidateCaptureIds[candidate.Word].Contains(candidate.CaptureSessionId)));
        }

        [Test]
        public void RefreshFailureKeepsPreviousImmutableViewReference()
        {
            CopyTree(PhysicalRoot(), tempRoot);
            var repository = NewRepository();
            repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            repository.TryCommitGuidedEnrollmentSeedQuorum("THANK_YOU");
            TrustedPersonalizationActiveViewService service = NewActiveViewService(repository);
            Assert.IsTrue(service.TryRefresh().Succeeded);
            TrustedPersonalizationActiveView previous = service.Current;
            File.AppendAllText(Path.Combine(CandidateDirectories(repository).First(), "checksum_manifest.json"), "tamper");
            TrustedActiveViewBuildResult failed = service.TryRefresh();
            Assert.IsFalse(failed.Succeeded);
            Assert.AreSame(previous, service.Current);
            Assert.IsFalse(failed.Replaced);
        }

        [Test]
        public void PostManifestUniqueReviewOnlySaveSucceedsWithoutChangingManifestAndRemainsUntrusted()
        {
            var repository = NewRepository();
            for (int sample = 1; sample <= 4; sample++)
                SaveReview(repository, "LOVE", "post-review-before-" + sample, sample,
                    0.60 - sample * 0.05, -sample * 0.02);
            GuidedEnrollmentSeedCommitResult committed = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            byte[] manifestBytes = File.ReadAllBytes(committed.ManifestPath);
            string manifestHash = Sha256(manifestBytes);
            DateTime manifestStamp = File.GetLastWriteTimeUtc(committed.ManifestPath);

            CandidateSaveResult later = SaveReview(
                repository, "LOVE", "post-review-later", 20, 0.90, -0.01);

            Assert.AreEqual(CandidateStorageStatus.Saved, later.Status);
            Assert.AreEqual(5, repository.Counts.ReviewOnly["LOVE"]);
            Assert.AreEqual(3, repository.TrustedInventory.CountForWord("LOVE"));
            Assert.IsFalse(repository.TrustedInventory.Candidates.Any(candidate =>
                candidate.CaptureSessionId == "post-review-later"));
            CollectionAssert.AreEqual(manifestBytes, File.ReadAllBytes(committed.ManifestPath));
            Assert.AreEqual(manifestHash, Sha256(File.ReadAllBytes(committed.ManifestPath)));
            Assert.AreEqual(manifestStamp, File.GetLastWriteTimeUtc(committed.ManifestPath));
            TrustedActiveViewBuildResult active = NewActiveViewService(repository).TryRefresh();
            Assert.IsTrue(active.Succeeded);
            Assert.IsFalse(active.ActiveView.CandidateCaptureIds["LOVE"].Contains("post-review-later"));
        }

        [Test]
        public void ExistingManifestAfterLaterReviewOnlyIsIdempotentNoWrite()
        {
            var repository = NewRepository();
            for (int sample = 1; sample <= 4; sample++)
                SaveReview(repository, "LOVE", "idempotent-before-" + sample, sample,
                    0.60 - sample * 0.05, -sample * 0.02);
            GuidedEnrollmentSeedCommitResult committed = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            byte[] before = File.ReadAllBytes(committed.ManifestPath);
            DateTime stamp = File.GetLastWriteTimeUtc(committed.ManifestPath);
            string[] selected = committed.SelectedCaptureSessionIds.ToArray();
            Assert.AreEqual(CandidateStorageStatus.Saved,
                SaveReview(repository, "LOVE", "idempotent-later", 20, 0.95, -0.01).Status);

            GuidedEnrollmentSeedCommitResult repeated = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");

            Assert.AreEqual(GuidedEnrollmentSeedCommitStatus.AlreadyCommitted, repeated.Status);
            CollectionAssert.AreEqual(selected, repeated.SelectedCaptureSessionIds);
            CollectionAssert.AreEqual(before, File.ReadAllBytes(committed.ManifestPath));
            Assert.AreEqual(stamp, File.GetLastWriteTimeUtc(committed.ManifestPath));
        }

        [Test]
        public void FrozenInputUnselectedCandidateTamperFailsClosed()
        {
            var repository = NewRepository();
            SaveReview(repository, "LOVE", "frozen-selected-1", 1, 0.40, -0.10);
            SaveReview(repository, "LOVE", "frozen-selected-2", 2, 0.30, -0.10);
            SaveReview(repository, "LOVE", "frozen-selected-3", 3, 0.20, -0.10);
            SaveReview(repository, "LOVE", "frozen-unselected", 4, 0.05, -0.10);
            GuidedEnrollmentSeedCommitResult committed = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            CollectionAssert.DoesNotContain(committed.SelectedCaptureSessionIds, "frozen-unselected");
            RewriteCandidateMarginKeepingRawRepositoryValid(repository, "frozen-unselected", 0.05, 0.04);

            InvalidDataException failure = Assert.Throws<InvalidDataException>(
                () => repository.ReloadTrustedAndVerify());
            StringAssert.Contains("frozen input", failure.Message);
        }

        [Test]
        public void PostManifestStrictEligibleOneTwoThreeUseStrictFirstAndGuidedOnlyAsFillers()
        {
            var repository = NewRepository();
            for (int sample = 1; sample <= 4; sample++)
                SaveReview(repository, "LOVE", "strict-priority-review-" + sample, sample,
                    0.60 - sample * 0.05, -sample * 0.02);
            GuidedEnrollmentSeedCommitResult committed = repository.TryCommitGuidedEnrollmentSeedQuorum("LOVE");
            byte[] manifestBytes = File.ReadAllBytes(committed.ManifestPath);
            DateTime manifestStamp = File.GetLastWriteTimeUtc(committed.ManifestPath);
            AssertTrustBasis(repository.ReloadTrustedAndVerify(), "LOVE", 0, 3);

            for (int strictCount = 1; strictCount <= 3; strictCount++)
            {
                CandidateSaveResult saved = SaveStrict(
                    repository, "LOVE", "strict-priority-" + strictCount, 20 + strictCount);
                Assert.AreEqual(CandidateStorageStatus.Saved, saved.Status, "strict save " + strictCount);
                AssertTrustBasis(repository.ReloadTrustedAndVerify(), "LOVE", strictCount, 3 - strictCount);
                CollectionAssert.AreEqual(manifestBytes, File.ReadAllBytes(committed.ManifestPath));
                Assert.AreEqual(manifestStamp, File.GetLastWriteTimeUtc(committed.ManifestPath));
            }
        }

        PersonalizationCandidateRepository NewRepository() =>
            new PersonalizationCandidateRepository("local_learner", tempRoot);

        PersonalizationCandidateRepository RepositoryWithReviewQuorum(string word)
        {
            var repository = NewRepository();
            for (int sample = 1; sample <= 3; sample++)
                SaveReview(repository, word, word.ToLowerInvariant() + "-review-" + sample, sample);
            return repository;
        }

        static CandidateSaveResult SaveReview(
            PersonalizationCandidateRepository repository,
            string word,
            string id,
            int variant,
            double closedSetMargin = 0.25,
            double thresholdMargin = -0.10) =>
            SaveReview(repository, Attempt(id, word, GuidedQuestOutcomeKind.LowConfidenceCorrect,
                variant, closedSetMargin: closedSetMargin, thresholdMargin: thresholdMargin));

        static CandidateSaveResult SaveReview(
            PersonalizationCandidateRepository repository,
            PracticeRecognitionAttempt attempt) =>
            repository.Save(new PersonalizationCandidateSaveRequest(
                attempt, PersonalizationCandidateEligibility.ReviewOnly,
                new DateTime(2026, 8, 19, 0, 0, 0, DateTimeKind.Utc)));

        static CandidateSaveResult SaveStrict(
            PersonalizationCandidateRepository repository,
            string word,
            string id,
            int variant) =>
            repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt(id, word, GuidedQuestOutcomeKind.StrictSuccess, variant),
                PersonalizationCandidateEligibility.Eligible,
                new DateTime(2026, 8, 19, 0, 0, 0, DateTimeKind.Utc)));

        static PracticeRecognitionAttempt Attempt(
            string sessionId,
            string expected,
            GuidedQuestOutcomeKind kind,
            int variant,
            string sourceHash = null,
            string logicalHash = null,
            string sequenceSampleId = null,
            double closedSetMargin = 0.25,
            double thresholdMargin = -0.10)
        {
            string predicted = kind == GuidedQuestOutcomeKind.WrongWord
                ? (expected == "GREETING" ? "LOVE" : "GREETING")
                : expected;
            bool strict = kind == GuidedQuestOutcomeKind.StrictSuccess;
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed,
                true,
                expected,
                predicted,
                strict,
                strict,
                strict,
                false,
                false);
            double best = 0.60;
            double second = best + closedSetMargin;
            double third = second + 0.20;
            double greeting = expected == "GREETING" ? best : expected == "LOVE" ? second : third;
            double love = expected == "LOVE" ? best : expected == "THANK_YOU" ? second : third;
            double thanks = expected == "THANK_YOU" ? best : expected == "GREETING" ? second : third;
            RecognitionSequence sequence = Sequence(
                sequenceSampleId ?? sessionId,
                variant,
                sourceHash ?? Sha256Text("source::" + sessionId + "::" + variant),
                logicalHash ?? Sha256Text("logical::" + sessionId + "::" + variant));
            return new PracticeRecognitionAttempt(
                sessionId, expected, predicted, PersonalLiveCaptureState.Completed,
                outcome, null, sequence,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                PersonalMultiPrototypeRuntimeProfileLoader.ProfileId,
                best + thresholdMargin,
                null, null,
                new PracticeRecognitionMetrics
                {
                    StrictAccepted = strict,
                    StrictExpectedWordClassAccept = strict,
                    StrictQuestSuccess = strict,
                    GreetingDistance = greeting,
                    LoveDistance = love,
                    ThankYouDistance = thanks,
                    BestDistance = best,
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
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Failed,
                false, expected, string.Empty, false, false, false, true, true);
            return new PracticeRecognitionAttempt(
                sessionId, expected, string.Empty, PersonalLiveCaptureState.Failed,
                outcome, null, null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                PersonalMultiPrototypeRuntimeProfileLoader.ProfileId,
                0d, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
        }

        static RecognitionSequence Sequence(string sampleId, int variant, string sourceHash, string logicalHash)
        {
            const int frames = 5;
            var coordinates = new double[frames * RecognitionSequence.RequiredLandmarkCount * 3];
            var mask = Enumerable.Repeat(true, frames * RecognitionSequence.RequiredLandmarkCount).ToArray();
            for (int index = 0; index < coordinates.Length; index++)
                coordinates[index] = variant * 0.01 + index * 0.001;
            return new RecognitionSequence(
                sampleId, "UNLABELED", "RUNTIME", "runtime", sourceHash, logicalHash,
                frames, RecognitionSequence.RequiredLandmarkCount, coordinates, mask);
        }

        static TrustedPersonalizationActiveViewService NewActiveViewService(
            ITrustedPersonalizationCandidateRepository repository)
        {
            RuntimeRecognitionService global = RuntimeRecognitionService.Load(
                GlobalTemplateDirectory, GlobalThresholdPath);
            return new TrustedPersonalizationActiveViewService(global, FixedProfileDirectory, repository);
        }

        static IEnumerable<string> CandidateDirectories(PersonalizationCandidateRepository repository) =>
            repository.TrustedInventory.RawCounts.TotalCandidateCount == 0
                ? Enumerable.Empty<string>()
                : Directory.GetDirectories(Path.Combine(repository.RootDirectory, repository.LearnerId, "candidates"),
                    "*", SearchOption.AllDirectories)
                    .Where(path => File.Exists(Path.Combine(path, "candidate_metadata.json")));

        static string ManifestPath(PersonalizationCandidateRepository repository, string word) =>
            Path.Combine(repository.RootDirectory, repository.LearnerId,
                ".guided_enrollment_seeds", word + ".json");

        static void AssertTrustBasis(
            TrustedCandidateInventory inventory,
            string word,
            int strictCount,
            int guidedCount)
        {
            Assert.AreEqual(3, inventory.CountForWord(word));
            Assert.AreEqual(strictCount, inventory.Candidates.Count(candidate =>
                candidate.Word == word &&
                candidate.TrustBasis == TrustedPersonalizationTrustBasis.StrictEligible));
            Assert.AreEqual(guidedCount, inventory.Candidates.Count(candidate =>
                candidate.Word == word &&
                candidate.TrustBasis == TrustedPersonalizationTrustBasis.GuidedLowConfidenceSeed));
        }

        static void RewriteCandidateMarginKeepingRawRepositoryValid(
            PersonalizationCandidateRepository repository,
            string captureSessionId,
            double oldMargin,
            double newMargin)
        {
            string directory = CandidateDirectories(repository).Single(path =>
                Path.GetFileName(path) == captureSessionId);
            string metadataPath = Path.Combine(directory, "candidate_metadata.json");
            byte[] oldMetadata = File.ReadAllBytes(metadataPath);
            string oldHash = Sha256(oldMetadata);
            string oldText = System.Text.Encoding.UTF8.GetString(oldMetadata);
            string oldValue = oldMargin.ToString(System.Globalization.CultureInfo.InvariantCulture);
            string newValue = newMargin.ToString(System.Globalization.CultureInfo.InvariantCulture);
            string newText = oldText.Replace(
                "\"closed_set_margin\": " + oldValue,
                "\"closed_set_margin\": " + newValue);
            Assert.AreNotEqual(oldText, newText, "candidate margin test mutation must be applied");
            byte[] newMetadata = new System.Text.UTF8Encoding(false).GetBytes(newText);
            string newHash = Sha256(newMetadata);
            File.WriteAllBytes(metadataPath, newMetadata);

            string checksumPath = Path.Combine(directory, "checksum_manifest.json");
            string checksum = File.ReadAllText(checksumPath)
                .Replace("\"size_bytes\": " + oldMetadata.Length,
                    "\"size_bytes\": " + newMetadata.Length)
                .Replace(oldHash, newHash);
            File.WriteAllText(checksumPath, checksum, new System.Text.UTF8Encoding(false));

            string indexPath = Path.Combine(
                repository.RootDirectory, repository.LearnerId, "candidate_index.json");
            string index = File.ReadAllText(indexPath).Replace(oldHash, newHash);
            File.WriteAllText(indexPath, index, new System.Text.UTF8Encoding(false));
        }

        static string PhysicalRoot()
        {
            string validation = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Validation");
            string[] matches = Directory.GetDirectories(validation,
                "WordsInSilence_step10k_v2_2_clean_enrollment_*", SearchOption.TopDirectoryOnly);
            Assert.AreEqual(1, matches.Length, "the approved v2.2 physical source root must be unique");
            return matches[0];
        }

        static void CopyTree(string source, string destination)
        {
            foreach (string directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(destination + directory.Substring(source.Length));
            Directory.CreateDirectory(destination);
            foreach (string file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            {
                string target = destination + file.Substring(source.Length);
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                File.Copy(file, target, false);
            }
        }

        static Dictionary<string, string> HashTree(string root)
        {
            var result = new Dictionary<string, string>(StringComparer.Ordinal);
            if (!Directory.Exists(root)) return result;
            foreach (string path in Directory.GetFiles(root, "*", SearchOption.AllDirectories))
                result[path.Substring(root.Length).Replace('\\', '/')] = Sha256(File.ReadAllBytes(path));
            return result;
        }

        static void AssertCounts(TrustedCandidateInventory inventory, int greeting, int love, int thanks)
        {
            Assert.AreEqual(greeting, inventory.CountForWord("GREETING"));
            Assert.AreEqual(love, inventory.CountForWord("LOVE"));
            Assert.AreEqual(thanks, inventory.CountForWord("THANK_YOU"));
        }

        static int CountOccurrences(string text, string value)
        {
            int count = 0;
            for (int index = 0; (index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0;
                 index += value.Length) count++;
            return count;
        }

        static string Sha256Text(string text) => Sha256(System.Text.Encoding.UTF8.GetBytes(text));

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create())
                return string.Concat(sha.ComputeHash(bytes).Select(value => value.ToString("x2")));
        }

        sealed class FakeAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Ready;
            public PracticeRecognitionAttempt Next;
            public PersonalLiveCaptureState CaptureState => state;
            public string CaptureSessionId => Next == null ? string.Empty : Next.CaptureSessionId;
            public string ActiveProfileId => PersonalMultiPrototypeRuntimeProfileLoader.ProfileId;
            public bool CandidateRouteActive => true;
            public bool SetExpectedWord(string word) => state == PersonalLiveCaptureState.Ready;
            public bool BeginCapture()
            {
                if (state != PersonalLiveCaptureState.Ready) return false;
                state = PersonalLiveCaptureState.Capturing;
                return true;
            }
            public PracticeRecognitionAttempt FinalizeCapture()
            {
                state = Next == null ? PersonalLiveCaptureState.Failed : Next.CaptureState;
                return Next;
            }
            public bool ResetCapture()
            {
                if (state != PersonalLiveCaptureState.Completed && state != PersonalLiveCaptureState.Failed)
                    return false;
                state = PersonalLiveCaptureState.Ready;
                return true;
            }
        }

        sealed class FakePromotionRepository :
            ITrustedPersonalizationCandidateRepository,
            IGuidedEnrollmentSeedPromotionRepository
        {
            public CandidateStorageStatus SaveStatus = CandidateStorageStatus.Saved;
            public int SaveCalls;
            public int CommitCalls;
            public string RootDirectory => "fake";
            public string LearnerId => "local_learner";
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;
            public TrustedCandidateInventory TrustedInventory => TrustedCandidateInventory.Empty;
            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public TrustedCandidateInventory ReloadTrustedAndVerify() => TrustedInventory;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                return new CandidateSaveResult(SaveStatus, string.Empty, SaveStatus.ToString(), Counts);
            }
            public GuidedEnrollmentSeedCommitResult TryCommitGuidedEnrollmentSeedQuorum(string word)
            {
                CommitCalls++;
                return new GuidedEnrollmentSeedCommitResult(
                    GuidedEnrollmentSeedCommitStatus.InsufficientCandidates,
                    word, string.Empty, "fake insufficient");
            }
        }
    }
}
