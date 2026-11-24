// M3_NPC_INPUT_N2 - EditMode tests for interaction-key ownership across the REAL NPC
// dialogue lifecycle. (N2T: test-only corrections; the N2 production patchset is unchanged.)
//
// These drive the actual production seams, not static booleans: a real
// InWorldGreetingNpcDialogueController configured through its existing ConfigureForTests
// with the same mockable interfaces the existing V2 suite uses, a real
// PetRobotWorldEntryController with a real PetRobotWorldEntryGate and a real
// PetRobotWorldHandoffChannel, and a recording IPetRobotWorldSceneTransition so a
// suppressed press is proven to request NO scene load. The dialogue is driven to
// QuestComplete through the real outcome path.
//
// Frames are injected through SetFrameSourceForTest, mirroring the existing
// SetTimeSourceForTest, so the one-frame input-consumption rule is tested deterministically
// with no sleeps and no future-frame waits. Production leaves both sources unset.
//
// Learner data: ConfigureForTests is given a NULL identity on purpose. The existing
// GetCurrentCandidateCounts returns CandidateCountSnapshot.Empty for a null identity, so no
// PersonalizationCandidateRepository is constructed and no candidate file is read while the
// input-only lifecycle is driven; the mock gate and runtime supply everything else. One
// honest caveat: Awake still runs the production BuildIdentity() before ConfigureForTests
// replaces it. That is pre-existing production behaviour shared with the current V2 suite,
// it is not introduced by this fixture, and nothing here reads or asserts any learner
// identifier, profile, count or recording.
//
// Outcome kinds: LowConfidenceCorrect is deliberately quest-completing in the installed
// ApplyOutcome (Step 10T), so the retryable branch below uses WrongWord, which is the kind
// that reaches Feedback. No production behaviour and no existing test is changed to fit an
// assertion here.
using System;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    [TestFixture]
    public sealed class PetRobotWorldNpcInputArbitrationN2Tests
    {
        GameObject npcRoot;
        GameObject npcTarget;
        GameObject entryRoot;
        GameObject player;
        GameObject pet;
        InWorldGreetingNpcDialogueController npcController;
        PetRobotWorldEntryController entryController;
        RecordingTransition transition;
        FakeGateVerifier gate;
        FakeRuntimeOwner runtime;
        FakeCameraController cameraController;
        FakePlayerLock playerLock;
        FakeNpcPresenter npcPresenter;
        float testTime;
        int testFrame;

        [SetUp]
        public void SetUp()
        {
            WorldInteractionArbiter.ResetForTests();
            testTime = 0f;
            testFrame = 100;

            player = new GameObject("Player");
            pet = new GameObject("Pet");
            pet.transform.position = player.transform.position;   // pet gate open

            npcRoot = new GameObject("NpcRoot");
            npcTarget = new GameObject("Npc");                    // owned, destroyed below
            npcController = npcRoot.AddComponent<InWorldGreetingNpcDialogueController>();
            gate = new FakeGateVerifier();
            runtime = new FakeRuntimeOwner();
            cameraController = new FakeCameraController();
            playerLock = new FakePlayerLock();
            npcPresenter = new FakeNpcPresenter();
            npcController.ConfigureForTests(
                npcTarget.transform, player.transform, 5f, gate,
                (PetRobotLearnerRootIdentity)null,     // Empty candidate counts, no disk read
                runtime, cameraController, playerLock, npcPresenter);
            npcController.SetTimeSourceForTest(() => testTime);
            npcController.SetFrameSourceForTest(() => testFrame);
            npcController.SetPlayerInsideGreetingTrigger(true);

            entryRoot = new GameObject("EntryRoot");
            entryController = entryRoot.AddComponent<PetRobotWorldEntryController>();
            transition = new RecordingTransition();
            entryController.ConfigureForTests(
                player.transform, pet.transform, "TestWorld", "TestPractice", 2.5f,
                new PetRobotWorldHandoffChannel(), transition, delegate { });
            entryController.RefreshInteractionRange();
            Assert.IsTrue(entryController.PlayerInRange,
                "precondition: the pet gate must be open, otherwise suppression is untested");
            Assert.AreEqual(1, WorldInteractionArbiter.ClaimantCount,
                "precondition: the real NPC controller must register itself in OnEnable");
        }

        [TearDown]
        public void TearDown()
        {
            if (npcRoot != null) UnityEngine.Object.DestroyImmediate(npcRoot);
            if (npcTarget != null) UnityEngine.Object.DestroyImmediate(npcTarget);
            if (entryRoot != null) UnityEngine.Object.DestroyImmediate(entryRoot);
            if (player != null) UnityEngine.Object.DestroyImmediate(player);
            if (pet != null) UnityEngine.Object.DestroyImmediate(pet);
            WorldInteractionArbiter.ResetForTests();
        }

        // ---- helpers that drive the REAL lifecycle ---------------------------

        void BeginRecognition()
        {
            if (npcController.State == InWorldDialogueState.QuestPrompt)
            {
                testTime += npcController.QuestPromptSeconds + 0.01f;
                npcController.DriveUpdate();
            }
        }

        string SimulateOutcome(TrustedHybridOutcomeKind kind)
        {
            string id = Guid.NewGuid().ToString("N");
            npcController.InjectOutcomeForTest(id, kind);
            npcController.DriveUpdate();
            testTime += npcController.AnalysisDisplaySeconds + 0.01f;
            npcController.DriveUpdate();
            return id;
        }

        void CompleteQuest()
        {
            gate.Ready = true;
            Assert.IsTrue(npcController.TryStartDialogue());
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);
            Assert.AreEqual(InWorldDialogueState.QuestComplete, npcController.State);
            Assert.IsTrue(npcController.ItemGranted);
        }

        void AssertPracticeSuppressed(string because)
        {
            Assert.IsFalse(entryController.TryEnterPractice(true), because);
            Assert.AreEqual(0, transition.RequestCount,
                "a suppressed press must request no scene load");
            Assert.IsFalse(entryController.TransitionRequested);
        }

        void AssertPracticeEntered()
        {
            Assert.IsTrue(entryController.TryEnterPractice(true));
            Assert.AreEqual(1, transition.RequestCount);
            Assert.AreEqual("TestPractice", transition.LastScene);
        }

        // ---- start, active, retry, completed-open ---------------------------

        [Test]
        public void OpenDialogue_AtStart_OwnsTheKey()
        {
            gate.Ready = true;
            Assert.IsTrue(npcController.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, npcController.State);
            Assert.IsTrue(npcController.DialogueOpen);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("an open dialogue must own the interaction key");
        }

        [Test]
        public void ActiveRecognition_OwnsTheKey()
        {
            gate.Ready = true;
            npcController.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, npcController.State);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("an active recognition must own the interaction key");
        }

        [Test]
        public void RetryFeedback_OwnsTheKey()
        {
            gate.Ready = true;
            npcController.TryStartDialogue();
            BeginRecognition();
            // WrongWord is the retryable branch. LowConfidenceCorrect is deliberately
            // quest-completing in the installed ApplyOutcome (Step 10T) and must not be
            // used here, and that production behaviour is not changed to suit this test.
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);
            Assert.AreEqual(InWorldDialogueState.Feedback, npcController.State);
            Assert.IsTrue(npcController.CanRetry);
            Assert.IsFalse(npcController.ItemGranted);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("a retryable feedback state must own the key");
        }

        [Test]
        public void CompletedButStillOpen_OwnsTheKey()
        {
            CompleteQuest();
            Assert.IsTrue(npcController.DialogueOpen,
                "QuestComplete has not been closed yet, so the interaction is still open");
            Assert.IsFalse(npcController.InteractionFinished);
            AssertPracticeSuppressed("an unclosed QuestComplete must own the key");
        }

        // ---- the same-frame close, both handler orders -----------------------

        [Test]
        public void CloseThenEntry_InTheSameFrame_CannotEnterPractice()
        {
            CompleteQuest();
            npcController.CloseDialogue();                 // the NPC handler ran first
            Assert.IsFalse(npcController.DialogueOpen);
            Assert.IsTrue(npcController.ConsumedInteractionThisFrame);
            Assert.AreEqual("npc_close_consumed_frame", npcController.ClaimReason);
            AssertPracticeSuppressed("the press that closed the dialogue must not also "
                                     + "enter practice in the same frame");
            Assert.AreEqual("npc_close_consumed_frame",
                            entryController.LastSuppressionReason);
        }

        [Test]
        public void EntryThenClose_InTheSameFrame_CannotEnterPractice()
        {
            CompleteQuest();
            AssertPracticeSuppressed("the entry controller ran first, while the dialogue "
                                     + "was still open");
            npcController.CloseDialogue();
            Assert.AreEqual(0, transition.RequestCount);
        }

        // ---- the N1 regression this proposal corrects ------------------------

        [Test]
        public void ClosedCompletedAndAway_OnALaterFrame_EntersPracticeNormally()
        {
            CompleteQuest();
            npcController.CloseDialogue();
            npcController.SetPlayerInsideGreetingTrigger(false);
            testFrame++;

            Assert.AreEqual(InWorldDialogueState.QuestComplete, npcController.State,
                "the completed quest state must NOT be reset to release the key");
            Assert.IsTrue(npcController.ItemGranted, "the granted item must be preserved");
            Assert.IsTrue(npcController.InteractionFinished);
            Assert.IsFalse(npcController.ClaimsInteractionInput,
                "a finished, closed interaction must release the interaction key");
            AssertPracticeEntered();
        }

        [Test]
        public void ClosedCompletedButStillAtTheNpc_AlsoReleasesTheKey()
        {
            CompleteQuest();
            npcController.CloseDialogue();
            testFrame++;                                   // player never left the trigger

            Assert.IsTrue(npcController.NpcInRange);
            Assert.IsFalse(npcController.ClaimsInteractionInput,
                "a finished NPC has nothing left to do with the key");
            AssertPracticeEntered();
        }

        [Test]
        public void RepeatedCloseWhileAlreadyClosed_DoesNotRenewSuppression()
        {
            CompleteQuest();
            npcController.CloseDialogue();
            testFrame++;
            npcController.CloseDialogue();                 // the DriveInput repeat path

            Assert.IsFalse(npcController.ConsumedInteractionThisFrame,
                "only a real close may consume a frame");
            Assert.IsFalse(npcController.ClaimsInteractionInput);
            AssertPracticeEntered();
        }

        // ---- idle ownership and ordinary pet entry ---------------------------

        [Test]
        public void IdleAtTheNpc_OwnsTheKeyAndNeverFabricatesSuccess()
        {
            Assert.AreEqual(InWorldDialogueState.Idle, npcController.State);
            Assert.IsTrue(npcController.NpcInRange);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("standing at the NPC gives the key to the NPC");
            Assert.IsFalse(npcController.ItemGranted);
            Assert.AreEqual(InWorldDialogueState.Idle, npcController.State);
        }

        [Test]
        public void IdleAtTheNpcWithAnUnreadyGate_StillSuppressesAndRefusesTruthfully()
        {
            gate.Ready = false;
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("a blocked NPC must not fall through to practice");
            Assert.IsFalse(npcController.TryStartDialogue(),
                "the enrollment gate must still refuse");
            Assert.AreEqual(InWorldDialogueState.Idle, npcController.State);
            Assert.IsFalse(npcController.ItemGranted);
        }

        [Test]
        public void IdleAwayFromTheNpc_EntersPracticeNormally()
        {
            npcController.SetPlayerInsideGreetingTrigger(false);
            Assert.IsFalse(npcController.ClaimsInteractionInput);
            AssertPracticeEntered();
        }

        [Test]
        public void NoClaimantRegistered_OrdinaryPetEntry()
        {
            WorldInteractionArbiter.ResetForTests();
            Assert.AreEqual(0, WorldInteractionArbiter.ClaimantCount);
            AssertPracticeEntered();
            Assert.AreEqual(string.Empty, entryController.LastSuppressionReason);
        }

        [Test]
        public void DestroyedNpc_IsPrunedAndDoesNotBlockEntry()
        {
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            UnityEngine.Object.DestroyImmediate(npcRoot);
            npcRoot = null;

            string reason;
            Assert.IsFalse(WorldInteractionArbiter.TryGetClaim(out reason),
                "a destroyed NPC must never hold the interaction key");
            AssertPracticeEntered();
        }

        [Test]
        public void DisabledNpc_UnregistersAndDoesNotBlockEntry()
        {
            npcRoot.SetActive(false);
            Assert.AreEqual(0, WorldInteractionArbiter.ClaimantCount,
                "OnDisable must unregister the claimant");
            AssertPracticeEntered();
        }

        // ---- the pure mapping, for completeness ------------------------------

        [Test]
        public void MappingReleasesTheKeyOnlyWhenNothingOwnsIt()
        {
            Assert.IsTrue(WorldInteractionArbiter.NpcOwnsInteractionKey(true, false, false, false));
            Assert.IsTrue(WorldInteractionArbiter.NpcOwnsInteractionKey(false, true, false, false));
            Assert.IsTrue(WorldInteractionArbiter.NpcOwnsInteractionKey(false, true, true, true),
                "a real close consumes its frame even after the interaction finished");
            Assert.IsFalse(WorldInteractionArbiter.NpcOwnsInteractionKey(false, true, true, false),
                "a finished interaction releases the key even inside the trigger");
            Assert.IsFalse(WorldInteractionArbiter.NpcOwnsInteractionKey(false, false, false, false));
        }

        // --- test doubles, shaped like the existing V2 suite's ---

        sealed class RecordingTransition : IPetRobotWorldSceneTransition
        {
            public int RequestCount { get; private set; }
            public string LastScene { get; private set; }

            public bool Request(string sceneName)
            {
                RequestCount++;
                LastScene = sceneName;
                return true;
            }
        }

        sealed class FakeGateVerifier : IQuestGateVerifier
        {
            public bool Ready;
            public string Failure = string.Empty;
            public string LastFailure => Failure;
            public bool VerifyOverallEnrollmentGateReady() => Ready;
        }

        sealed class FakeRuntimeOwner : IInWorldRecognitionRuntimeOwner
        {
            public long Generation = 1;
            public bool Disposed;
            public bool FirstObservationReady;

            public bool IsActive => !Disposed;
            public long FrozenGeneration => Generation;
            public PetRobotSignPracticeSession Session => null;
            public PetRobotNeutralAutoCaptureCoordinator Coordinator => null;
            public string InitializationFailure => string.Empty;

            public bool RequestAttempt() => true;
            public bool RetrySession() => true;

            public InWorldRecognitionReadinessSnapshot SnapshotReadiness()
            {
                return new InWorldRecognitionReadinessSnapshot(
                    true, true, false, FirstObservationReady,
                    RawMediaPipeNeutralAutoSegmenterState.Inactive, string.Empty);
            }

            public void Dispose() { Disposed = true; }
        }

        sealed class FakeCameraController : IInWorldDialogueCameraController
        {
            public bool Saved;
            public bool Restored;
            public void SaveState() { Saved = true; Restored = false; }
            public void EnterDialogueCamera() { }
            public void RestoreState() { Restored = true; }
        }

        sealed class FakePlayerLock : IInWorldDialoguePlayerLock
        {
            public bool Locked;
            public void Lock() { Locked = true; }
            public void Unlock() { Locked = false; }
        }

        sealed class FakeNpcPresenter : IInWorldNpcPresenter
        {
            public int GreetingCount;
            public int ConfusedCount;
            public int GiveItemCount;
            public void TriggerGreeting() { GreetingCount++; }
            public void TriggerConfused() { ConfusedCount++; }
            public void TriggerGiveItem() { GiveItemCount++; }
        }
    }
}
