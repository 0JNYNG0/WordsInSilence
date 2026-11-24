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
    public sealed class TrustedPersonalizedEnrollmentAndHybridRuntimeV1Tests
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
            tempRoot = Path.Combine(Path.GetTempPath(), "wis_step10k_v2_1_" + Guid.NewGuid().ToString("N"));
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(tempRoot)) Directory.Delete(tempRoot, true);
        }

        [Test]
        public void ProductionCurrentEightCandidates_ReadOnlyTrustedCountsAreTwoTwoOne()
        {
            string root = PersonalizationCandidateRepository.DefaultRootDirectory;
            Dictionary<string, string> before = HashTree(root);
            var repository = new PersonalizationCandidateRepository("local_learner");
            TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();
            Assert.AreEqual(8, inventory.RawCounts.TotalCandidateCount);
            Assert.AreEqual(2, inventory.CountForWord("GREETING"));
            Assert.AreEqual(2, inventory.CountForWord("LOVE"));
            Assert.AreEqual(1, inventory.CountForWord("THANK_YOU"));
            Assert.IsFalse(inventory.OverallEnrollmentGateReady);
            CollectionAssert.AreEquivalent(before, HashTree(root));
        }

        [Test]
        public void InjectedCleanRoot_StartsZeroZeroZeroWithoutTouchingProductionBytes()
        {
            Dictionary<string, string> productionBefore =
                HashTree(PersonalizationCandidateRepository.DefaultRootDirectory);
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            AssertCounts(repository.TrustedInventory, 0, 0, 0);
            CollectionAssert.AreEquivalent(
                productionBefore, HashTree(PersonalizationCandidateRepository.DefaultRootDirectory));
        }

        [Test]
        public void CleanRoot_EachWordZeroToOneToTwoToThreeAndOverallGateFalseToTrue()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            foreach (string word in new[] { "GREETING", "LOVE", "THANK_YOU" })
            {
                for (int sample = 1; sample <= 3; sample++)
                {
                    Assert.AreEqual(CandidateStorageStatus.Saved, SaveStrict(repository, word, sample).Status);
                    Assert.AreEqual(sample, repository.TrustedInventory.CountForWord(word));
                    if (word != "THANK_YOU" || sample < 3)
                        Assert.IsFalse(repository.TrustedInventory.OverallEnrollmentGateReady);
                }
            }
            AssertCounts(repository.TrustedInventory, 3, 3, 3);
            Assert.IsTrue(repository.TrustedInventory.OverallEnrollmentGateReady);
        }

        [Test]
        public void SameCleanRoot_ReconstructionRestoresPartialAndCompletedCounts()
        {
            var first = new PersonalizationCandidateRepository("local_learner", tempRoot);
            SaveStrict(first, "GREETING", 1);
            SaveStrict(first, "LOVE", 1);
            var partial = new PersonalizationCandidateRepository("local_learner", tempRoot);
            AssertCounts(partial.TrustedInventory, 1, 1, 0);
            for (int sample = 2; sample <= 3; sample++)
            {
                SaveStrict(partial, "GREETING", sample);
                SaveStrict(partial, "LOVE", sample);
            }
            for (int sample = 1; sample <= 3; sample++) SaveStrict(partial, "THANK_YOU", sample);
            var completed = new PersonalizationCandidateRepository("local_learner", tempRoot);
            AssertCounts(completed.TrustedInventory, 3, 3, 3);
            Assert.IsTrue(completed.TrustedInventory.OverallEnrollmentGateReady);
            var reentered = new PetRobotSignPracticeSession(new FakeAdapter(), completed);
            Assert.AreEqual(PetRobotPracticeState.Completed, reentered.State);
            Assert.AreEqual(0, reentered.CurrentScoreableAttemptCount);
            Assert.AreEqual(0, reentered.CurrentConsecutiveTechnicalFailureCount);
        }

        [Test]
        public void DifferentCleanValidationRoot_DoesNotImportEarlierValidationCounts()
        {
            var first = new PersonalizationCandidateRepository("local_learner", Path.Combine(tempRoot, "a"));
            SaveStrict(first, "GREETING", 1);
            var second = new PersonalizationCandidateRepository("local_learner", Path.Combine(tempRoot, "b"));
            AssertCounts(second.TrustedInventory, 0, 0, 0);
        }

        [Test]
        public void Repository_StrictVerificationIncludesWisrcbManifestMetadataAndReviewOnlyIsolation()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            SaveStrict(repository, "GREETING", 1);
            CandidateSaveResult review = repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt("review-love", "LOVE", GuidedQuestOutcomeKind.LowConfidenceCorrect, 9),
                PersonalizationCandidateEligibility.ReviewOnly,
                DateTime.UtcNow));
            Assert.AreEqual(CandidateStorageStatus.Saved, review.Status);
            TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();
            AssertCounts(inventory, 1, 0, 0);
            Assert.AreEqual(1, inventory.RawCounts.ReviewOnly["LOVE"]);
            Assert.AreEqual(1, inventory.ExcludedDuplicateOrUntrustedCount);
            Assert.AreEqual("WISRCB01", System.Text.Encoding.ASCII.GetString(
                File.ReadAllBytes(Path.Combine(inventory.Candidates[0].CandidateDirectory,
                    "normalized_sequence.wisrq")), 0, 8));
        }

        [Test]
        public void Repository_DuplicateCaptureSequenceSourceOrLogicalIdentityNeverCountsTwice()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            PracticeRecognitionAttempt original = Attempt("original", "GREETING", GuidedQuestOutcomeKind.StrictSuccess, 1);
            Assert.AreEqual(CandidateStorageStatus.Saved, repository.Save(Request(original)).Status);
            PracticeRecognitionAttempt duplicatePayload = Attempt(
                "new-capture-id", "GREETING", GuidedQuestOutcomeKind.StrictSuccess, 1,
                original.NormalizedSequence.SourceNpzSha256,
                original.NormalizedSequence.LogicalPayloadSha256);
            Assert.AreEqual(CandidateStorageStatus.Duplicate, repository.Save(Request(duplicatePayload)).Status);
            Assert.AreEqual(CandidateStorageStatus.Duplicate, repository.Save(Request(original)).Status);
            Assert.AreEqual(1, repository.TrustedInventory.CountForWord("GREETING"));
        }

        [Test]
        public void Session_RestoresPersonalizedWordAndSkipsToFirstIncompleteWord()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            for (int sample = 1; sample <= 3; sample++) SaveStrict(repository, "GREETING", sample);
            var session = new PetRobotSignPracticeSession(new FakeAdapter(), repository);
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.Personalized,
                session.GetEnrollmentState("GREETING"));
            Assert.AreEqual("LOVE", session.CurrentWord);
            Assert.IsTrue(session.IsWordComplete("GREETING"));
        }

        [Test]
        public void Session_StrictUniqueSavedCompletesExactlyAtThirdTrustedSample()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            for (int sample = 1; sample <= 3; sample++)
            {
                adapter.Next = Attempt("session-g-" + sample, "GREETING", GuidedQuestOutcomeKind.StrictSuccess, sample);
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());
                Assert.AreEqual(sample, session.CurrentTrustedSampleCount);
                Assert.AreEqual(sample == 3, session.IsCurrentWordComplete);
                if (sample < 3) Assert.IsTrue(session.Retry());
            }
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.Personalized, session.CurrentEnrollmentState);
            Assert.AreEqual(3, session.CurrentScoreableAttemptCount);
        }

        [TestCase(CandidateStorageStatus.Failed)]
        [TestCase(CandidateStorageStatus.Duplicate)]
        public void Session_StrictStorageFailureOrDuplicateNeverCompletes(CandidateStorageStatus status)
        {
            var repository = new FakeTrustedRepository { NextStatus = status };
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.Next = Attempt("strict-noncounting-" + status, "GREETING", GuidedQuestOutcomeKind.StrictSuccess, 1);
            session.StartExpression();
            session.FinishExpression();
            Assert.AreEqual(1, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(0, session.CurrentTrustedSampleCount);
            Assert.IsFalse(session.IsCurrentWordComplete);
        }

        [Test]
        public void Session_LowConfidenceReviewOnlyDoesNotCompleteOrIncreaseTrustedCount()
        {
            var repository = new FakeTrustedRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.Next = Attempt("low", "GREETING", GuidedQuestOutcomeKind.LowConfidenceCorrect, 1);
            session.StartExpression();
            session.FinishExpression();
            Assert.AreEqual(PersonalizationCandidateEligibility.ReviewOnly, repository.LastEligibility);
            Assert.AreEqual(1, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(0, session.CurrentTrustedSampleCount);
            Assert.IsTrue(session.CanRetry);
        }

        [Test]
        public void Session_WrongWordDoesNotSaveAndCanRetryBeforeAttemptTen()
        {
            var repository = new FakeTrustedRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.Next = Attempt("wrong", "GREETING", GuidedQuestOutcomeKind.WrongWord, 1);
            session.StartExpression();
            session.FinishExpression();
            Assert.AreEqual(0, repository.SaveCalls);
            Assert.AreEqual(1, session.CurrentScoreableAttemptCount);
            Assert.IsTrue(session.CanRetry);
        }

        [Test]
        public void Session_TenthScoreableAttemptEndsRoundIncompleteWithoutCompletingGate()
        {
            var repository = new FakeTrustedRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            for (int attempt = 1; attempt <= 10; attempt++)
            {
                adapter.Next = Attempt("wrong-" + attempt, "GREETING", GuidedQuestOutcomeKind.WrongWord, attempt);
                session.StartExpression();
                session.FinishExpression();
                if (attempt < 10) Assert.IsTrue(session.Retry());
            }
            Assert.AreEqual(10, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete,
                session.CurrentEnrollmentState);
            Assert.IsFalse(session.IsCurrentWordComplete);
            Assert.IsFalse(session.OverallEnrollmentGateReady);
            Assert.IsTrue(session.CanContinueWithNeedsPractice);
        }

        [Test]
        public void Session_TechnicalFailureExcludedFromAttemptsThreeConsecutiveStopsAndScoreableResets()
        {
            var repository = new FakeTrustedRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.Next = TechnicalAttempt("tech-1", "GREETING");
            session.StartExpression(); session.FinishExpression();
            Assert.AreEqual(0, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(1, session.CurrentConsecutiveTechnicalFailureCount);
            session.Retry();
            adapter.Next = Attempt("wrong-resets", "GREETING", GuidedQuestOutcomeKind.WrongWord, 1);
            session.StartExpression(); session.FinishExpression();
            Assert.AreEqual(0, session.CurrentConsecutiveTechnicalFailureCount);
            session.Retry();
            for (int failure = 1; failure <= 3; failure++)
            {
                adapter.Next = TechnicalAttempt("tech-stop-" + failure, "GREETING");
                session.StartExpression(); session.FinishExpression();
                if (failure < 3) session.Retry();
            }
            Assert.AreEqual(1, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(3, session.CurrentConsecutiveTechnicalFailureCount);
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.TechnicalStop,
                session.CurrentEnrollmentState);
            Assert.IsFalse(session.IsCurrentWordComplete);
            Assert.IsFalse(session.OverallEnrollmentGateReady);
            Assert.IsTrue(session.CanContinueWithNeedsPractice);
            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual("LOVE", session.CurrentWord);
            Assert.IsFalse(session.IsWordComplete("GREETING"));
            Assert.IsFalse(session.OverallEnrollmentGateReady);
        }

        [Test]
        public void Session_DuplicateCallbackDoesNotDoubleCountAttemptSampleOrCompletion()
        {
            var repository = new FakeTrustedRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.Next = Attempt("same", "GREETING", GuidedQuestOutcomeKind.StrictSuccess, 1);
            session.StartExpression(); session.FinishExpression();
            Assert.AreEqual(1, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(1, session.CurrentTrustedSampleCount);
            Assert.IsTrue(session.Retry());
            adapter.Next = Attempt("same", "GREETING", GuidedQuestOutcomeKind.StrictSuccess, 2);
            session.StartExpression(); session.FinishExpression();
            Assert.AreEqual(1, session.CurrentScoreableAttemptCount);
            Assert.AreEqual(1, session.CurrentTrustedSampleCount);
            Assert.AreEqual(1, repository.SaveCalls);
        }

        [Test]
        public void StoppedRoundCanAdvanceAndExitButNeverMarksWordCompleteOrGateReady()
        {
            var repository = new FakeTrustedRepository();
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            for (int attempt = 1; attempt <= 10; attempt++)
            {
                adapter.Next = Attempt("g-" + attempt, "GREETING", GuidedQuestOutcomeKind.WrongWord, attempt);
                session.StartExpression(); session.FinishExpression();
                if (attempt < 10) session.Retry();
            }
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete,
                session.CurrentEnrollmentState);
            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual("LOVE", session.CurrentWord);
            Assert.IsFalse(session.IsWordComplete("GREETING"));
            Assert.IsFalse(session.OverallEnrollmentGateReady);

            for (int attempt = 1; attempt <= 10; attempt++)
            {
                adapter.Next = Attempt("l-" + attempt, "LOVE", GuidedQuestOutcomeKind.WrongWord, attempt);
                session.StartExpression(); session.FinishExpression();
                if (attempt < 10) session.Retry();
            }
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete,
                session.CurrentEnrollmentState);
            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual("THANK_YOU", session.CurrentWord);
            Assert.IsFalse(session.IsWordComplete("LOVE"));
            Assert.IsFalse(session.OverallEnrollmentGateReady);

            for (int attempt = 1; attempt <= 10; attempt++)
            {
                adapter.Next = Attempt("t-" + attempt, "THANK_YOU", GuidedQuestOutcomeKind.WrongWord, attempt);
                session.StartExpression(); session.FinishExpression();
                if (attempt < 10) session.Retry();
            }
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete,
                session.CurrentEnrollmentState);
            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
            Assert.IsFalse(session.IsWordComplete("GREETING"));
            Assert.IsFalse(session.IsWordComplete("LOVE"));
            Assert.IsFalse(session.IsWordComplete("THANK_YOU"));
            Assert.IsFalse(session.AllWordsComplete);
            Assert.IsFalse(session.OverallEnrollmentGateReady);
            Assert.IsTrue(session.ExitPractice());
            Assert.AreEqual(PetRobotPracticeState.Exited, session.State);

            var reentered = new PetRobotSignPracticeSession(new FakeAdapter(), repository);
            Assert.AreEqual(PetRobotPracticeState.Ready, reentered.State);
            Assert.AreEqual("GREETING", reentered.CurrentWord);
            Assert.AreEqual(PetRobotPersonalizationEnrollmentState.Pending,
                reentered.CurrentEnrollmentState);
            Assert.IsFalse(reentered.IsCurrentWordComplete);
            Assert.IsFalse(reentered.OverallEnrollmentGateReady);
            Assert.IsTrue(reentered.StartExpression());
        }

        [Test]
        public void ThreeWordCleanSessionRequiresThreeEachBeforeOverallGateAndCompletion()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            var adapter = new FakeAdapter();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            int ordinal = 0;
            foreach (string word in new[] { "GREETING", "LOVE", "THANK_YOU" })
            {
                for (int sample = 1; sample <= 3; sample++)
                {
                    ordinal++;
                    adapter.Next = Attempt("all-" + ordinal, word, GuidedQuestOutcomeKind.StrictSuccess, ordinal);
                    session.StartExpression(); session.FinishExpression();
                    if (sample < 3) session.Retry();
                }
                Assert.AreEqual(PetRobotPersonalizationEnrollmentState.Personalized,
                    session.GetEnrollmentState(word));
                session.ContinueToNextWord();
            }
            Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
            Assert.IsTrue(session.AllWordsComplete);
            Assert.IsTrue(session.OverallEnrollmentGateReady);
        }

        [Test]
        public void ActiveView_ExcludesAllLiveCandidatesBelowPerWordQuorum()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            SaveStrict(repository, "GREETING", 1);
            SaveStrict(repository, "GREETING", 2);
            var service = NewActiveViewService(repository);
            Assert.IsTrue(service.TryRefresh().Succeeded);
            Assert.AreEqual(0, service.Current.CandidateCaptureIds["GREETING"].Count);
            Assert.AreEqual(3, service.Current.Profile.Prototypes["GREETING"].Count);
        }

        [Test]
        public void ActiveView_AtQuorumIncludesOnlyTrustedEligibleAndKeepsFrozenThreshold()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            for (int sample = 1; sample <= 3; sample++) SaveStrict(repository, "GREETING", sample);
            repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt("review", "GREETING", GuidedQuestOutcomeKind.LowConfidenceCorrect, 20),
                PersonalizationCandidateEligibility.ReviewOnly, DateTime.UtcNow));
            PersonalMultiPrototypeRuntimeProfile fixedProfile =
                PersonalMultiPrototypeRuntimeProfileLoader.Load(FixedProfileDirectory);
            var service = NewActiveViewService(repository);
            Assert.IsTrue(service.TryRefresh().Succeeded);
            Assert.AreEqual(3, service.Current.CandidateCaptureIds["GREETING"].Count);
            Assert.AreEqual(6, service.Current.Profile.Prototypes["GREETING"].Count);
            Assert.AreEqual(fixedProfile.Thresholds["GREETING"],
                service.Current.Profile.Thresholds["GREETING"], 0d);
        }

        [Test]
        public void ActiveView_BuildFailureKeepsPreviousReferenceAndNeverPartiallyPromotes()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            for (int sample = 1; sample <= 3; sample++) SaveStrict(repository, "GREETING", sample);
            var service = NewActiveViewService(repository);
            Assert.IsTrue(service.TryRefresh().Succeeded);
            TrustedPersonalizationActiveView previous = service.Current;
            string metadata = Directory.GetFiles(tempRoot, "candidate_metadata.json", SearchOption.AllDirectories)[0];
            File.AppendAllText(metadata, "tamper");
            TrustedActiveViewBuildResult failed = service.TryRefresh();
            Assert.IsFalse(failed.Succeeded);
            Assert.IsFalse(failed.Replaced);
            Assert.AreSame(previous, service.Current);
            Assert.AreEqual(3, service.Current.CandidateCaptureIds["GREETING"].Count);
        }

        [Test]
        public void ActiveViewAndGlobalAssetsRemainByteIdenticalAfterBuild()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            for (int sample = 1; sample <= 3; sample++) SaveStrict(repository, "LOVE", sample);
            Dictionary<string, string> fixedBefore = HashTree(FixedProfileDirectory);
            Dictionary<string, string> globalBefore = HashTree(Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1"));
            NewActiveViewService(repository).TryRefresh();
            CollectionAssert.AreEquivalent(fixedBefore, HashTree(FixedProfileDirectory));
            CollectionAssert.AreEquivalent(globalBefore, HashTree(Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1")));
        }

        [Test]
        public void RuntimeSessionSnapshotRemainsImmutableAcrossRepositoryChangeAndRefresh()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            var service = NewActiveViewService(repository);
            TrustedHybridRuntimeSession acquired = service.AcquireSession();
            long generation = acquired.ActiveViewGeneration;
            for (int sample = 1; sample <= 3; sample++) SaveStrict(repository, "THANK_YOU", sample);
            service.TryRefresh();
            Assert.AreEqual(generation, acquired.ActiveViewGeneration);
            Assert.AreNotSame(acquired.ActiveView, service.Current);
            Assert.IsFalse(acquired.RuntimeObservationCanMutateEnrollment);
            Assert.IsFalse(service.RuntimeObservationCanMutateEnrollment);
        }

        [Test]
        public void HybridRuntimeFailureIsTypedTechnicalAndCannotWriteEnrollment()
        {
            var repository = new PersonalizationCandidateRepository("local_learner", tempRoot);
            var service = NewActiveViewService(repository);
            TrustedHybridRuntimeSession runtime = service.AcquireSession();
            TrustedHybridRecognitionResult result = runtime.Recognize("missing-query", null, "GREETING");
            Assert.AreEqual(TrustedHybridOutcomeKind.TechnicalFailure, result.Kind);
            Assert.IsNotEmpty(result.FailureType);
            AssertCounts(repository.TrustedInventory, 0, 0, 0);
            Assert.IsFalse(runtime.RuntimeObservationCanMutateEnrollment);
        }

        [TestCase(true, false, true, false, "GREETING", "GREETING", TrustedHybridOutcomeKind.StrictSuccess)]
        [TestCase(true, false, false, false, "GREETING", "LOVE", TrustedHybridOutcomeKind.LowConfidenceCorrect)]
        [TestCase(false, false, true, false, "LOVE", "GREETING", TrustedHybridOutcomeKind.LowConfidenceCorrect)]
        [TestCase(true, false, false, true, "GREETING", "LOVE", TrustedHybridOutcomeKind.WrongWord)]
        [TestCase(false, true, true, false, "LOVE", "GREETING", TrustedHybridOutcomeKind.WrongWord)]
        [TestCase(false, false, false, false, "LOVE", "THANK_YOU", TrustedHybridOutcomeKind.WrongWord)]
        public void HybridOutcomeMatrix_IsConservative(
            bool globalExpected,
            bool globalConflict,
            bool personalExpected,
            bool personalConflict,
            string globalWord,
            string personalWord,
            TrustedHybridOutcomeKind expected)
        {
            Assert.AreEqual(expected, TrustedHybridOutcomePolicy.Evaluate(
                globalExpected, globalConflict, personalExpected, personalConflict,
                globalWord, personalWord));
        }

        // R4: SOURCE-ONLY wiring check, rescoped to the PRACTICE construction path.
        //
        // The previous form banned TrustedPersonalizationActiveViewService from the ENTIRE
        // bootstrap, but the bootstrap has legitimately owned it inside the QUEST-only helper
        // CreateQuestHybridSession since before F28. The intended guarantee is unchanged: the
        // practice enrollment route must not construct a personalized legacy hybrid or a
        // self-reinforcing evaluator. That guarantee is now checked where it belongs, on both
        // practice construction branches (legacy and full-hand).
        //
        // This reads SOURCE TEXT only. It is not runtime proof; the behavioural coverage lives
        // in the Session_* tests above and in the F28 store/session tests.
        [Test]
        public void PracticeConstructionSelectsNoPersonalizedLegacyHybrid_SourceOnly()
        {
            string bootstrap = Source("PetRobotPracticeVerticalSliceBootstrap.cs");
            string session = Source("PetRobotSignPracticeSession.cs");

            // "else if (IsQuestMode)" occurs twice in the bootstrap - once in the repository
            // selection and once in the session construction - so the slice is anchored first
            // on the unique session-construction marker, and the legacy quest branch is then
            // resolved unambiguously INSIDE that slice.
            const string sessionConstructionMarker = "if (IsQuestMode && fullHandStore != null)";
            int sessionConstructionStart =
                bootstrap.IndexOf(sessionConstructionMarker, StringComparison.Ordinal);
            Assert.GreaterOrEqual(sessionConstructionStart, 0,
                "source marker not found: " + sessionConstructionMarker);
            Assert.AreEqual(sessionConstructionStart,
                bootstrap.LastIndexOf(sessionConstructionMarker, StringComparison.Ordinal),
                "source marker must occur exactly once: " + sessionConstructionMarker);
            string sessionConstruction = bootstrap.Substring(sessionConstructionStart);

            // the legacy quest branch is the ONLY caller of the quest hybrid helper
            string legacyQuestBranch =
                UniqueBlockAfter(sessionConstruction, "else if (IsQuestMode)");
            Assert.AreEqual(1, Occurrences(bootstrap, "CreateQuestHybridSession(repository)"),
                "the quest hybrid helper must be called exactly once in the bootstrap");
            StringAssert.Contains("CreateQuestHybridSession(repository)", legacyQuestBranch);

            // the two PRACTICE construction branches select neither the quest hybrid helper
            // nor any personalized legacy hybrid/active-view type
            string fullHandPracticeBranch =
                UniqueBlockAfter(bootstrap, "else if (fullHandStore != null)");
            foreach (string banned in new[]
            {
                "CreateQuestHybridSession",
                "TrustedHybridRecognitionSessionAdapter",
                "TrustedPersonalizationActiveViewService",
                "TrustedHybridRuntimeSession",
            })
                StringAssert.DoesNotContain(banned, fullHandPracticeBranch);

            const string legacyPracticeConstruction =
                "session = new PetRobotSignPracticeSession(adapter, repository, null, candidateCameraRoute);";
            Assert.AreEqual(1, Occurrences(bootstrap, legacyPracticeConstruction),
                "the legacy practice construction must appear exactly once and pass no hybrid session");

            // the full-hand quest branch also constructs no legacy hybrid
            string fullHandQuestBranch =
                UniqueBlockAfter(bootstrap, "if (IsQuestMode && fullHandStore != null)");
            StringAssert.DoesNotContain("CreateQuestHybridSession", fullHandQuestBranch);
            StringAssert.DoesNotContain("TrustedHybridRecognitionSessionAdapter", fullHandQuestBranch);

            // the personalized active-view service appears ONLY inside the quest helper body
            string questHelper = UniqueBlockAfter(
                bootstrap, "TrustedHybridRuntimeSession CreateQuestHybridSession(");
            Assert.AreEqual(
                Occurrences(bootstrap, "TrustedPersonalizationActiveViewService"),
                Occurrences(questHelper, "TrustedPersonalizationActiveViewService"),
                "the personalized active-view service must stay inside the quest-only helper");

            // the session itself still constructs no legacy hybrid runtime of its own
            StringAssert.DoesNotContain("TrustedHybridRuntimeSession", session);
            StringAssert.Contains("new PersonalizationCandidateRepository(learnerProfileId)", bootstrap);
            StringAssert.Contains("new PersonalLivePracticeRecognitionAdapter(harness)", bootstrap);
        }

        [Test]
        public void PlayerFacingResetProfileSelectionAndManualFinishWereNotAddedToGuidedFlow()
        {
            string combined = string.Join("\n", new[]
            {
                Source("GuidedSignPracticePresentationController.cs"),
                Source("MoaStoryGuidedPracticeFlow.cs"),
                Source("PetRobotPracticeVerticalSliceBootstrap.cs"),
            });
            StringAssert.DoesNotContain("새로 시작하기", combined);
            StringAssert.DoesNotContain("사용자 선택", combined);
            StringAssert.DoesNotContain("데이터 초기화", combined);
            StringAssert.Contains("HideLegacyNormalStoryControls();", Source("GuidedSignPracticePresentationController.cs"));
            StringAssert.Contains("finishButton.gameObject.SetActive(false);", Source("GuidedSignPracticePresentationController.cs"));
            StringAssert.Contains("RequestCoordinatorAttempt", Source("GuidedSignPracticePresentationController.cs"));
            StringAssert.Contains("검증된 개인 표본", Source("GuidedSignPracticePresentationController.cs"));
            StringAssert.Contains("연습 시도", Source("GuidedSignPracticePresentationController.cs"));
            StringAssert.Contains("기술 오류", Source("MoaStoryGuidedPracticeFlow.cs"));
        }

        [Test]
        public void LegacyPersonalizationTypesRemainDisconnectedFromCurrentBootstrap()
        {
            string bootstrap = Source("PetRobotPracticeVerticalSliceBootstrap.cs");
            foreach (string legacy in new[]
                     {
                         "PersonalizationService", "PersonalDataRepository", "PersonalSignEnrollment",
                         "SignPracticeController", "HandPose", "SignFeatureSet",
                     })
                StringAssert.DoesNotContain(legacy, bootstrap);
        }

        /// <summary>
        /// R4 source-only helper. Returns the balanced { ... } block that follows the ONE
        /// occurrence of <paramref name="marker"/>, asserting the marker is unique so a renamed
        /// or duplicated branch fails loudly instead of silently matching nothing. It is a
        /// plain brace matcher over source text, not a parser, and it is used only on blocks
        /// that contain no braces inside string literals.
        /// </summary>
        static string UniqueBlockAfter(string source, string marker)
        {
            int first = source.IndexOf(marker, StringComparison.Ordinal);
            Assert.GreaterOrEqual(first, 0, "source marker not found: " + marker);
            Assert.AreEqual(first, source.LastIndexOf(marker, StringComparison.Ordinal),
                "source marker must occur exactly once: " + marker);
            int open = source.IndexOf('{', first + marker.Length);
            Assert.GreaterOrEqual(open, 0, "no block follows the marker: " + marker);
            int depth = 0;
            for (int index = open; index < source.Length; index++)
            {
                if (source[index] == '{') depth++;
                else if (source[index] == '}')
                {
                    depth--;
                    if (depth == 0) return source.Substring(open, index - open + 1);
                }
            }
            Assert.Fail("unbalanced block after the marker: " + marker);
            return string.Empty;
        }

        static int Occurrences(string source, string value)
        {
            int count = 0;
            int at = source.IndexOf(value, StringComparison.Ordinal);
            while (at >= 0)
            {
                count++;
                at = source.IndexOf(value, at + value.Length, StringComparison.Ordinal);
            }
            return count;
        }

        static TrustedPersonalizationActiveViewService NewActiveViewService(
            ITrustedPersonalizationCandidateRepository repository)
        {
            RuntimeRecognitionService global = RuntimeRecognitionService.Load(
                GlobalTemplateDirectory, GlobalThresholdPath);
            return new TrustedPersonalizationActiveViewService(global, FixedProfileDirectory, repository);
        }

        static CandidateSaveResult SaveStrict(
            PersonalizationCandidateRepository repository,
            string word,
            int ordinal)
        {
            return repository.Save(Request(Attempt(
                word.ToLowerInvariant() + "-trusted-" + ordinal,
                word,
                GuidedQuestOutcomeKind.StrictSuccess,
                ordinal)));
        }

        static PersonalizationCandidateSaveRequest Request(PracticeRecognitionAttempt attempt)
        {
            return new PersonalizationCandidateSaveRequest(
                attempt, PersonalizationCandidateEligibility.Eligible,
                new DateTime(2026, 8, 18, 0, 0, 0, DateTimeKind.Utc));
        }

        static PracticeRecognitionAttempt Attempt(
            string sessionId,
            string expected,
            GuidedQuestOutcomeKind kind,
            int variant,
            string sourceHash = null,
            string logicalHash = null)
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
            RecognitionSequence sequence = Sequence(
                sessionId,
                variant,
                sourceHash ?? Sha256Text("source::" + sessionId + "::" + variant),
                logicalHash ?? Sha256Text("logical::" + sessionId + "::" + variant));
            return new PracticeRecognitionAttempt(
                sessionId,
                expected,
                predicted,
                PersonalLiveCaptureState.Completed,
                outcome,
                null,
                sequence,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                PersonalMultiPrototypeRuntimeProfileLoader.ProfileId,
                0.5113059473720765,
                null,
                null,
                new PracticeRecognitionMetrics
                {
                    StrictAccepted = strict,
                    StrictExpectedWordClassAccept = strict,
                    StrictQuestSuccess = strict,
                    GreetingDistance = expected == "GREETING" ? 0.2 : 0.7,
                    LoveDistance = expected == "LOVE" ? 0.2 : 0.7,
                    ThankYouDistance = expected == "THANK_YOU" ? 0.2 : 0.7,
                    BestDistance = 0.2,
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
                false,
                expected,
                string.Empty,
                false,
                false,
                false,
                true,
                true);
            return new PracticeRecognitionAttempt(
                sessionId, expected, string.Empty, PersonalLiveCaptureState.Failed,
                outcome, null, null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                PersonalMultiPrototypeRuntimeProfileLoader.ProfileId,
                0d, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
        }

        static RecognitionSequence Sequence(
            string sessionId,
            int variant,
            string sourceHash,
            string logicalHash)
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

        static void AssertCounts(TrustedCandidateInventory inventory, int greeting, int love, int thanks)
        {
            Assert.AreEqual(greeting, inventory.CountForWord("GREETING"));
            Assert.AreEqual(love, inventory.CountForWord("LOVE"));
            Assert.AreEqual(thanks, inventory.CountForWord("THANK_YOU"));
        }

        static string Source(string fileName)
        {
            return File.ReadAllText(Path.Combine(
                Application.dataPath, "_Project", "Scripts", "Practice",
                "PetRobotVerticalSlice", fileName));
        }

        static Dictionary<string, string> HashTree(string root)
        {
            var result = new Dictionary<string, string>(StringComparer.Ordinal);
            if (!Directory.Exists(root)) return result;
            foreach (string path in Directory.GetFiles(root, "*", SearchOption.AllDirectories))
                result[path.Substring(root.Length).Replace('\\', '/')] = Sha256(File.ReadAllBytes(path));
            return result;
        }

        static string Sha256Text(string value)
        {
            return Sha256(System.Text.Encoding.UTF8.GetBytes(value));
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create())
                return string.Concat(sha.ComputeHash(bytes).Select(item => item.ToString("x2")));
        }

        sealed class FakeAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Ready;
            public PracticeRecognitionAttempt Next;
            public PersonalLiveCaptureState CaptureState => state;
            public string CaptureSessionId => Next == null ? string.Empty : Next.CaptureSessionId;
            public string ActiveProfileId => PersonalMultiPrototypeRuntimeProfileLoader.ProfileId;
            public bool CandidateRouteActive => true;
            public bool SetExpectedWord(string word) =>
                state == PersonalLiveCaptureState.Ready &&
                PersonalLiveRecognitionController.IsSupportedWord(word);
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

        sealed class FakeTrustedRepository : ITrustedPersonalizationCandidateRepository
        {
            readonly Dictionary<string, int> trusted = Counts(0);
            readonly Dictionary<string, int> eligible = Counts(0);
            readonly Dictionary<string, int> review = Counts(0);
            public CandidateStorageStatus NextStatus = CandidateStorageStatus.Saved;
            public int SaveCalls;
            public PersonalizationCandidateEligibility LastEligibility;
            public string RootDirectory => "fake-clean-root";
            public string LearnerId => "local_learner";
            public CandidateCountSnapshot Counts => new CandidateCountSnapshot(eligible, review);
            public TrustedCandidateInventory TrustedInventory => new TrustedCandidateInventory(
                Counts, trusted, Array.Empty<TrustedPersonalizationCandidate>(), review.Values.Sum());
            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public TrustedCandidateInventory ReloadTrustedAndVerify() => TrustedInventory;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                LastEligibility = request.Eligibility;
                if (NextStatus == CandidateStorageStatus.Saved)
                {
                    if (request.Eligibility == PersonalizationCandidateEligibility.Eligible)
                    {
                        eligible[request.Attempt.ExpectedWord]++;
                        trusted[request.Attempt.ExpectedWord]++;
                    }
                    else review[request.Attempt.ExpectedWord]++;
                }
                return new CandidateSaveResult(NextStatus, "fake", NextStatus.ToString(), Counts);
            }
        }

        static Dictionary<string, int> Counts(int value)
        {
            return new Dictionary<string, int>(StringComparer.Ordinal)
            {
                { "GREETING", value }, { "LOVE", value }, { "THANK_YOU", value },
            };
        }
    }
}
