using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
#endif

namespace WordsInSilence.Tests.EditMode
{
    [TestFixture]
    public sealed class InWorldGreetingNpcDialogueV2Tests
    {
        GameObject root;
        InWorldGreetingNpcDialogueController controller;
        Transform npc;
        Transform player;
        FakeGateVerifier gate;
        FakeRuntimeOwner runtime;
        FakeCameraController camera;
        FakePlayerLock playerLock;
        FakeNpcPresenter npcPresenter;
        float testTime;

        [SetUp]
        public void SetUp()
        {
            testTime = 0f;
            root = new GameObject("TestRoot");
            controller = root.AddComponent<InWorldGreetingNpcDialogueController>();
            npc = new GameObject("NPC").transform;
            player = new GameObject("Player").transform;
            npc.position = Vector3.zero;
            player.position = Vector3.forward * 2f;
            gate = new FakeGateVerifier();
            runtime = new FakeRuntimeOwner();
            camera = new FakeCameraController();
            playerLock = new FakePlayerLock();
            npcPresenter = new FakeNpcPresenter();

            controller.ConfigureForTests(
                npc, player, 5f, gate,
                new PetRobotLearnerRootIdentity("test_learner"),
                runtime, camera, playerLock, npcPresenter);

            controller.SetTimeSourceForTest(() => testTime);
            controller.SetPlayerInsideGreetingTrigger(true);
        }

        [TearDown]
        public void TearDown()
        {
            if (root != null) UnityEngine.Object.DestroyImmediate(root);
            if (npc != null) UnityEngine.Object.DestroyImmediate(npc.gameObject);
            if (player != null) UnityEngine.Object.DestroyImmediate(player.gameObject);
        }

        // === Behavior tests §1–§34 (39 tests) ===

        // §1: Gate must be ready
        [Test]
        public void Dialogue_RequiresGateReady_BeforeAttempt()
        {
            gate.Ready = false;
            Assert.IsFalse(controller.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.Idle, controller.State);
        }

        // §2: Gate insufficient/sufficient/exception
        [Test]
        public void Gate_Insufficient_Blocks()
        {
            gate.Ready = false;
            gate.Failure = "GREETING 2, LOVE 2, THANK_YOU 1";
            Assert.IsFalse(controller.TryStartDialogue());
        }

        [Test]
        public void Gate_ThreeThreeThree_Allows()
        {
            gate.Ready = true;
            Assert.IsTrue(controller.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, controller.State);
        }

        [Test]
        public void Gate_Exception_FailClosed_NoThrow_NoCameraMutation()
        {
            gate.ThrowOnVerify = true;
            Assert.IsFalse(controller.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.Idle, controller.State);
            Assert.IsFalse(camera.Saved, "camera must not be saved on gate exception");
            Assert.IsFalse(playerLock.Locked, "player must not be locked on gate exception");
        }

        // §3: No scene transition, handoff, watcher, transition request
        [Test]
        public void Source_NoSceneManagerLoadScene()
        {
            string source = File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsFalse(source.Contains("SceneManager.LoadScene"),
                "field path must not call SceneManager.LoadScene");
        }

        [Test]
        public void Source_NoHandoffChannelBeginGreetingQuest()
        {
            string source = File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsFalse(source.Contains("BeginGreetingQuest"),
                "field path must not call BeginGreetingQuest");
        }

        [Test]
        public void Source_NoPracticeReturnWatcher()
        {
            string source = File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsFalse(source.Contains("PetRobotPracticeReturnWatcher"),
                "field path must not use return watcher");
        }

        [Test]
        public void Source_NoTransitionRequest()
        {
            string source = File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsFalse(source.Contains("transition.Request"),
                "field path must not call transition.Request");
        }

        // §4: First recognition calls RequestAttempt exactly once
        [Test]
        public void FirstRecognition_CallsRequestAttemptOnce()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(1, runtime.RequestAttemptCount);
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
        }

        // §5: RequestAttempt failure -> fail-closed
        [Test]
        public void RequestAttemptFailure_FailClosed()
        {
            gate.Ready = true;
            runtime.RequestAttemptResult = false;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(runtime.Disposed, "dead runtime must be disposed");
        }

        // §6: Retry calls RetrySession then RequestAttempt exactly once each
        [Test]
        public void Retry_CallsRetrySessionThenRequestAttemptOnce()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            runtime.RequestAttemptCount = 0;
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(1, runtime.RetrySessionCount);
            Assert.AreEqual(1, runtime.RequestAttemptCount);
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
        }

        // §7: Init failure retry disposes and recreates runtime
        [Test]
        public void InitFailureRetry_DisposesAndRecreatesRuntime()
        {
            var failingRuntime = new FakeRuntimeOwner();
            failingRuntime.FailureMessage = "simulated init failure";
            var workingRuntime = new FakeRuntimeOwner();

            int callCount = 0;
            controller.SetRuntimeFactoryForTest(() =>
            {
                callCount++;
                return callCount == 1 ? failingRuntime : workingRuntime;
            });

            gate.Ready = true;
            controller.TryStartDialogue();

            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.IsTrue(failingRuntime.Disposed, "failed runtime must be disposed");

            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
            Assert.AreEqual(2, callCount, "factory must be called twice (fail + recreate)");
            Assert.AreEqual(1, workingRuntime.RequestAttemptCount);
        }

        // §8: Frozen generation maintained across retry
        //
        // R4: the retry driver is WrongWord. LowConfidenceCorrect is a quest-COMPLETING
        // outcome in the unchanged Step 10T policy, so it leaves QuestComplete and cannot be
        // retried; that legacy completion is asserted on its own below.
        [Test]
        public void FrozenGeneration_MaintainedAcrossRetry()
        {
            gate.Ready = true;
            runtime.Generation = 42;
            controller.TryStartDialogue();
            BeginRecognition();
            long gen1 = controller.FrozenGeneration;

            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);
            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(gen1, controller.FrozenGeneration);
            Assert.AreEqual(42, controller.FrozenGeneration);
        }

        // §9: Null/no outcome produces no reward/state completion
        [Test]
        public void NoOutcome_NoRewardNoCompletion()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            controller.DriveUpdate();
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
            Assert.IsFalse(controller.ItemGranted);
            Assert.AreEqual(InWorldDialogueOutcome.None, controller.LastOutcome);
        }

        // §10-§13: Four outcome kinds
        [Test]
        public void StrictSuccess_QuestComplete_RewardOnce()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);

            Assert.AreEqual(InWorldDialogueState.QuestComplete, controller.State);
            Assert.IsTrue(controller.ItemGranted);
            Assert.AreEqual(1, npcPresenter.GiveItemCount);
            Assert.IsFalse(controller.CanRetry);
        }

        // R4 rename of LowConfidenceCorrect_Confused_Retry_NoReward.
        //
        // This asserts the UNCHANGED Step 10T legacy policy: an injected legacy
        // LowConfidenceCorrect completes the quest and grants the item exactly once, while the
        // outcome LABEL stays LowConfidenceCorrect. It is deliberately NOT a statement about
        // the full-hand common-only Before authority, whose accepted-only route is unchanged
        // and covered by its own tests.
        [Test]
        public void LegacyLowConfidenceCorrect_CompletesQuestOnce_WithUnchangedLabel()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.LowConfidenceCorrect);

            Assert.AreEqual(InWorldDialogueState.QuestComplete, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.LowConfidenceCorrect, controller.LastOutcome,
                "the outcome label must not be promoted to StrictSuccess");
            Assert.AreEqual(TrustedHybridOutcomeKind.LowConfidenceCorrect,
                controller.LastAppliedOutcomeKind,
                "the applied canonical kind must stay LowConfidenceCorrect");
            Assert.IsTrue(controller.ItemGranted);
            Assert.AreEqual(1, npcPresenter.GiveItemCount, "the item is granted exactly once");
            Assert.AreEqual(0, npcPresenter.ConfusedCount, "a completing outcome is not confused");
            Assert.IsFalse(controller.CanRetry,
                "CanRetry requires the Feedback state, and a completed quest is not Feedback");
            Assert.IsFalse(controller.TryRetry());
        }

        [Test]
        public void WrongWord_Confused_Retry_NoReward()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.WrongWord, controller.LastOutcome);
            Assert.AreEqual(1, npcPresenter.ConfusedCount);
            Assert.IsFalse(controller.ItemGranted);
            Assert.AreEqual(0, npcPresenter.GiveItemCount);
            Assert.IsTrue(controller.CanRetry);
        }

        [Test]
        public void TechnicalFailure_NoScore_Retry_NoReward()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.TechnicalFailure);

            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsFalse(controller.ItemGranted);
            Assert.AreEqual(0, npcPresenter.GiveItemCount);
            Assert.IsTrue(controller.CanRetry);
        }

        // §14: Duplicate attempt one-shot consumption
        //
        // R4: the first outcome is WrongWord, a non-rewarding verdict that stays in Feedback,
        // so the one-shot rule is observed without depending on the legacy completing outcome.
        [Test]
        public void DuplicateAttempt_OneShot()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            string id = SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);
            int confusedAfterFirst = npcPresenter.ConfusedCount;

            controller.InjectOutcomeForTest(id, TrustedHybridOutcomeKind.StrictSuccess);
            controller.DriveUpdate();
            testTime += controller.AnalysisDisplaySeconds + 0.01f;
            controller.DriveUpdate();
            Assert.AreEqual(confusedAfterFirst, npcPresenter.ConfusedCount,
                "duplicate must not re-trigger");
            Assert.AreEqual(InWorldDialogueOutcome.WrongWord, controller.LastOutcome,
                "outcome must not change on duplicate");
            Assert.IsFalse(controller.ItemGranted);
            Assert.AreEqual(0, npcPresenter.GiveItemCount,
                "a duplicate attempt id must never grant the item");
        }

        // §15: Candidate write 0 -- source verification
        [Test]
        public void Source_NoCandidateWrite()
        {
            string source = File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsFalse(source.Contains("SaveCandidate"),
                "controller must not call SaveCandidate");
            Assert.IsFalse(source.Contains("WriteCandidate"),
                "controller must not call WriteCandidate");
            Assert.IsFalse(source.Contains("StoreCandidate"),
                "controller must not call StoreCandidate");
        }

        // §16: Candidate write 0 -- runtime owner tracks no writes
        [Test]
        public void CandidateRepository_NoWriteDuringFieldOutcome()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(0, runtime.CandidateWriteCount, "no writes before outcome");
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);
            Assert.AreEqual(0, runtime.CandidateWriteCount, "no writes after StrictSuccess");
        }

        // §17: Retry uses same frozen generation
        [Test]
        public void Retry_SameFrozenGeneration()
        {
            gate.Ready = true;
            runtime.Generation = 99;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
            Assert.AreEqual(99, controller.FrozenGeneration);
        }

        // §18-§20: Camera/player state restored
        [Test]
        public void StateRestored_OnClose_AfterSuccess()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            Assert.IsTrue(camera.Saved);
            Assert.IsTrue(playerLock.Locked);

            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);
            controller.CloseDialogue();

            Assert.IsTrue(camera.Restored);
            Assert.IsFalse(playerLock.Locked);
        }

        [Test]
        public void StateRestored_OnClose_AfterCancel()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            controller.CloseDialogue();

            Assert.IsTrue(camera.Restored);
            Assert.IsFalse(playerLock.Locked);
        }

        [Test]
        public void StateRestored_OnClose_AfterTechnicalFailure()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.TechnicalFailure);
            controller.CloseDialogue();

            Assert.IsTrue(camera.Restored);
            Assert.IsFalse(playerLock.Locked);
        }

        // §21: Runtime disposed on close
        [Test]
        public void RuntimeOwner_DisposedOnClose()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.IsTrue(runtime.IsActive);
            controller.CloseDialogue();
            Assert.IsTrue(runtime.Disposed);
        }

        // §22: ItemGranted persists
        [Test]
        public void ItemGranted_PersistsAfterClose()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);
            controller.CloseDialogue();
            Assert.IsTrue(controller.ItemGranted);
            Assert.IsFalse(controller.CanStartDialogue);
        }

        // §23: Trigger outside — cannot start
        [Test]
        public void TriggerOutside_CannotStartDialogue()
        {
            gate.Ready = true;
            controller.SetPlayerInsideGreetingTrigger(false);
            Assert.IsFalse(controller.CanStartDialogue);
            Assert.IsFalse(controller.TryStartDialogue());
        }

        // §24: Dialogue start triggers NPC greeting
        [Test]
        public void DialogueStart_TriggersNpcGreeting()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            Assert.AreEqual(1, npcPresenter.GreetingCount);
        }

        // §25: NullAttemptId outcome still works
        [Test]
        public void NullAttemptId_GeneratesId()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            controller.InjectOutcomeForTest(null, TrustedHybridOutcomeKind.StrictSuccess);
            controller.DriveUpdate();
            testTime += controller.AnalysisDisplaySeconds + 0.01f;
            controller.DriveUpdate();
            Assert.AreEqual(InWorldDialogueState.QuestComplete, controller.State);
            Assert.IsTrue(controller.ItemGranted);
            Assert.IsFalse(string.IsNullOrEmpty(controller.LastAttemptId));
        }

        // === v2.2 New tests: Retry dead-owner recovery ===

        // §26: Active retry -- RequestAttempt false disposes owner
        [Test]
        public void ActiveRetry_RequestAttemptFalse_DisposesOwner()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            runtime.RequestAttemptResult = false;
            Assert.IsFalse(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(runtime.Disposed, "owner must be disposed on active retry RequestAttempt failure");
        }

        // §27: Active retry -- RetrySession false disposes owner
        [Test]
        public void ActiveRetry_RetrySessionFalse_DisposesOwner()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            runtime.RetrySessionResult = false;
            Assert.IsFalse(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(runtime.Disposed, "owner must be disposed on RetrySession failure");
        }

        // §28: Active retry -- RetrySession exception disposes owner
        [Test]
        public void ActiveRetry_RetrySessionException_DisposesOwner()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            runtime.ThrowOnRetrySession = true;
            Assert.IsFalse(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(runtime.Disposed, "owner must be disposed on RetrySession exception");
        }

        // §29: Active retry -- RequestAttempt exception disposes owner
        [Test]
        public void ActiveRetry_RequestAttemptException_DisposesOwner()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            runtime.ThrowOnRequestAttempt = true;
            Assert.IsFalse(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(runtime.Disposed, "owner must be disposed on RequestAttempt exception");
        }

        // §30: Dead retry -- factory returns null fails closed
        [Test]
        public void DeadRetry_FactoryNull_FailsClosed()
        {
            var failingRuntime = new FakeRuntimeOwner();
            failingRuntime.FailureMessage = "init fail";
            int callCount = 0;
            controller.SetRuntimeFactoryForTest(() =>
            {
                callCount++;
                return callCount == 1 ? failingRuntime : null;
            });

            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();

            Assert.IsFalse(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
        }

        // §31: Dead retry -- recreated runtime RequestAttempt exception disposes
        [Test]
        public void DeadRetry_RequestAttemptException_Disposes()
        {
            var failingRuntime = new FakeRuntimeOwner();
            failingRuntime.FailureMessage = "init fail";
            var throwingRuntime = new FakeRuntimeOwner();
            throwingRuntime.ThrowOnRequestAttempt = true;
            int callCount = 0;
            controller.SetRuntimeFactoryForTest(() =>
            {
                callCount++;
                return callCount == 1 ? failingRuntime : throwingRuntime;
            });

            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();

            Assert.IsFalse(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(throwingRuntime.Disposed, "recreated runtime must be disposed");
        }

        // §32: Initial -- factory returns null fails closed
        [Test]
        public void Initial_FactoryNull_FailsClosed()
        {
            controller.SetRuntimeFactoryForTest(() => null);
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
        }

        // §33: Initial -- RequestAttempt exception disposes
        [Test]
        public void Initial_RequestAttemptException_Disposes()
        {
            var throwingRuntime = new FakeRuntimeOwner();
            throwingRuntime.ThrowOnRequestAttempt = true;
            controller.SetRuntimeFactoryForTest(() => throwingRuntime);

            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(throwingRuntime.Disposed, "runtime must be disposed on RequestAttempt exception");
        }

        // §34: Dead retry -- next R after dispose creates new runtime
        [Test]
        public void DeadRetry_NextR_CreatesNewRuntime()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            runtime.RetrySessionResult = false;
            Assert.IsFalse(controller.TryRetry());
            Assert.IsTrue(runtime.Disposed);

            var newRuntime = new FakeRuntimeOwner();
            controller.SetRuntimeFactoryForTest(() => newRuntime);
            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
            Assert.AreEqual(1, newRuntime.RequestAttemptCount);
        }

        // === Trigger-authoritative proximity tests §T1-§T8 ===

        // §T1: Trigger outside even when distance near — cannot start
        [Test]
        public void TriggerOutside_EvenWhenDistanceNear_CannotStart()
        {
            gate.Ready = true;
            player.position = npc.position;
            controller.SetPlayerInsideGreetingTrigger(false);
            Assert.IsFalse(controller.NpcInRange, "trigger-only: distance irrelevant");
            Assert.IsFalse(controller.CanStartDialogue);
            Assert.IsFalse(controller.TryStartDialogue());
        }

        // §T2: Trigger inside even when distance far — can start
        [Test]
        public void TriggerInside_EvenWhenDistanceFar_CanStart()
        {
            gate.Ready = true;
            player.position = Vector3.forward * 9999f;
            controller.SetPlayerInsideGreetingTrigger(true);
            Assert.IsTrue(controller.NpcInRange, "trigger-only: distance irrelevant");
            Assert.IsTrue(controller.CanStartDialogue);
            Assert.IsTrue(controller.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, controller.State);
        }

        // §T3: Relay — first enter sets true
        [Test]
        public void Relay_FirstEnter_SetsTrue()
        {
            var relayGo = CreateRelayFixture(out var relay, out var ctrl);
            try
            {
                var colliderA = CreateTaggedCollider("Player");
                RelayEnter(relay, colliderA);
                Assert.IsTrue(ctrl.NpcInRange, "first player enter must set trigger true");
                UnityEngine.Object.DestroyImmediate(colliderA.gameObject);
            }
            finally { DestroyRelayFixture(relayGo); }
        }

        // §T4: Two player colliders — one exit keeps true
        [Test]
        public void Relay_TwoColliders_OneExit_StillTrue()
        {
            var relayGo = CreateRelayFixture(out var relay, out var ctrl);
            try
            {
                var colA = CreateTaggedCollider("Player");
                var colB = CreateTaggedCollider("Player");
                RelayEnter(relay, colA);
                RelayEnter(relay, colB);
                Assert.IsTrue(ctrl.NpcInRange);

                RelayExit(relay, colA);
                Assert.IsTrue(ctrl.NpcInRange, "one collider still inside");

                RelayExit(relay, colB);
                Assert.IsFalse(ctrl.NpcInRange, "last collider exited");

                UnityEngine.Object.DestroyImmediate(colA.gameObject);
                UnityEngine.Object.DestroyImmediate(colB.gameObject);
            }
            finally { DestroyRelayFixture(relayGo); }
        }

        // §T5: Duplicate enter and unknown exit are safe
        [Test]
        public void Relay_DuplicateEnter_UnknownExit_Safe()
        {
            var relayGo = CreateRelayFixture(out var relay, out var ctrl);
            try
            {
                var col = CreateTaggedCollider("Player");
                RelayEnter(relay, col);
                RelayEnter(relay, col);
                Assert.IsTrue(ctrl.NpcInRange);

                var unknown = CreateTaggedCollider("Player");
                RelayExit(relay, unknown);
                Assert.IsTrue(ctrl.NpcInRange, "unknown collider exit must not affect state");

                RelayExit(relay, col);
                Assert.IsFalse(ctrl.NpcInRange);

                UnityEngine.Object.DestroyImmediate(col.gameObject);
                UnityEngine.Object.DestroyImmediate(unknown.gameObject);
            }
            finally { DestroyRelayFixture(relayGo); }
        }

        // §T6: OnDisable clears and sets false
        //
        // R4: the handler is invoked directly, because EditMode never runs the enable/disable
        // lifecycle for this component. After the handler, RE-ENTERING THE SAME collider must
        // set the trigger true again, which is only possible if the previous membership was
        // actually cleared.
        [Test]
        public void Relay_OnDisable_ClearsAndSetsFalse()
        {
            var relayGo = CreateRelayFixture(out var relay, out var ctrl);
            try
            {
                var col = CreateTaggedCollider("Player");
                RelayEnter(relay, col);
                Assert.IsTrue(ctrl.NpcInRange);

                RelayDisable(relay);
                Assert.IsFalse(ctrl.NpcInRange, "OnDisable must set trigger false");

                RelayEnter(relay, col);
                Assert.IsTrue(ctrl.NpcInRange,
                    "re-entering the same collider proves the previous membership was cleared");

                UnityEngine.Object.DestroyImmediate(col.gameObject);
            }
            finally { DestroyRelayFixture(relayGo); }
        }

        // §T7: Active dialogue — trigger exit then retry/close still works
        [Test]
        public void ActiveDialogue_TriggerExit_RetryCloseStillWorks()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            controller.SetPlayerInsideGreetingTrigger(false);
            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);

            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);

            controller.CloseDialogue();
            Assert.IsTrue(camera.Restored);
            Assert.IsFalse(playerLock.Locked);
        }

        // §T8: StrictSuccess → QuestComplete → close → restore
        [Test]
        public void StrictSuccess_Close_PlayerCameraRestore()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);

            Assert.AreEqual(InWorldDialogueState.QuestComplete, controller.State);
            Assert.IsTrue(controller.ItemGranted);
            controller.CloseDialogue();
            Assert.IsTrue(camera.Restored);
            Assert.IsFalse(playerLock.Locked);
        }

        // §T9: Child collider with Player-tagged parent hierarchy triggers relay
        [Test]
        public void Relay_ChildColliderWithPlayerParent_EnterExitWorks()
        {
            var relayGo = CreateRelayFixture(out var relay, out var ctrl);
            try
            {
                var parentGo = new GameObject("PlayerRoot");
                parentGo.tag = "Player";
                var childGo = new GameObject("ChildCollider");
                childGo.transform.SetParent(parentGo.transform);
                var childCol = childGo.AddComponent<BoxCollider>();

                RelayEnter(relay, childCol);
                Assert.IsTrue(ctrl.NpcInRange, "child collider under Player-tagged parent must set trigger true");

                RelayExit(relay, childCol);
                Assert.IsFalse(ctrl.NpcInRange, "child collider exit must set trigger false");

                UnityEngine.Object.DestroyImmediate(parentGo);
            }
            finally { DestroyRelayFixture(relayGo); }
        }

        // §T10: Non-Player hierarchy collider is ignored by relay
        [Test]
        public void Relay_NonPlayerHierarchyCollider_Ignored()
        {
            var relayGo = CreateRelayFixture(out var relay, out var ctrl);
            try
            {
                var parentGo = new GameObject("EnemyRoot");
                parentGo.tag = "Untagged";
                var childGo = new GameObject("EnemyChild");
                childGo.transform.SetParent(parentGo.transform);
                var childCol = childGo.AddComponent<BoxCollider>();

                RelayEnter(relay, childCol);
                Assert.IsFalse(ctrl.NpcInRange, "non-Player hierarchy must not set trigger true");

                UnityEngine.Object.DestroyImmediate(parentGo);
            }
            finally { DestroyRelayFixture(relayGo); }
        }

        // === §35: Actual session quest mode no-write — sentinel + full-tree invariant ===
        [Test]
        public void ActualSession_QuestMode_AllOutcomes_NoWrite()
        {
            string testRoot = Path.Combine(Path.GetTempPath(),
                "WIS_v2_3b_test_" + Guid.NewGuid().ToString("N"));
            try
            {
                Directory.CreateDirectory(testRoot);

                string sentinelPath = Path.Combine(testRoot, "sentinel.txt");
                File.WriteAllText(sentinelPath, "no-write-sentinel");

                string initialSnapshot = ComputeDirectorySnapshot(testRoot);
                int initialCandidateCount = new PersonalizationCandidateRepository(
                    "test_learner", testRoot).Counts.TotalCandidateCount;

                // --- Part A: Controller-level no-write (all 4 outcome kinds) ---
                foreach (TrustedHybridOutcomeKind kind in new[]
                {
                    TrustedHybridOutcomeKind.StrictSuccess,
                    TrustedHybridOutcomeKind.LowConfidenceCorrect,
                    TrustedHybridOutcomeKind.WrongWord,
                    TrustedHybridOutcomeKind.TechnicalFailure,
                })
                {
                    var go = new GameObject("CtrlTest_" + kind);
                    var ctrl = go.AddComponent<InWorldGreetingNpcDialogueController>();
                    var g = new FakeGateVerifier { Ready = true };
                    var r = new FakeRuntimeOwner();
                    var c = new FakeCameraController();
                    var pl = new FakePlayerLock();
                    var np = new FakeNpcPresenter();

                    ctrl.ConfigureForTests(npc, player, 5f, g,
                        new PetRobotLearnerRootIdentity("test_learner", testRoot),
                        r, c, pl, np);
                    ctrl.SetTimeSourceForTest(() => testTime);
                    ctrl.SetPlayerInsideGreetingTrigger(true);

                    ctrl.TryStartDialogue();
                    testTime += ctrl.QuestPromptSeconds + 0.01f;
                    ctrl.DriveUpdate();
                    ctrl.InjectOutcomeForTest(Guid.NewGuid().ToString("N"), kind);
                    ctrl.DriveUpdate();
                    testTime += ctrl.AnalysisDisplaySeconds + 0.01f;
                    ctrl.DriveUpdate();
                    ctrl.CloseDialogue();
                    UnityEngine.Object.DestroyImmediate(go);
                }

                string snapshotAfterPartA = ComputeDirectorySnapshot(testRoot);
                Assert.AreEqual(initialSnapshot, snapshotAfterPartA,
                    "full-tree must be invariant after controller-level 4 outcomes");

                // --- Part B: Session-level with all 5 outcome paths ---

                DriveSessionOutcome(testRoot, null, null);

                DriveSessionOutcome(testRoot,
                    GuidedQuestOutcomeKind.TechnicalFailure,
                    TrustedHybridOutcomeKind.TechnicalFailure);

                DriveSessionOutcome(testRoot,
                    GuidedQuestOutcomeKind.StrictSuccess,
                    TrustedHybridOutcomeKind.StrictSuccess);

                DriveSessionOutcome(testRoot,
                    GuidedQuestOutcomeKind.LowConfidenceCorrect,
                    TrustedHybridOutcomeKind.LowConfidenceCorrect);

                DriveSessionOutcome(testRoot,
                    GuidedQuestOutcomeKind.WrongWord,
                    TrustedHybridOutcomeKind.WrongWord);

                string finalSnapshot = ComputeDirectorySnapshot(testRoot);
                Assert.AreEqual(initialSnapshot, finalSnapshot,
                    "full-tree must be invariant after all session outcomes");

                int finalCandidateCount = new PersonalizationCandidateRepository(
                    "test_learner", testRoot).Counts.TotalCandidateCount;
                Assert.AreEqual(initialCandidateCount, finalCandidateCount,
                    "candidate count must remain unchanged");
            }
            finally
            {
                if (Directory.Exists(testRoot))
                    Directory.Delete(testRoot, true);
            }
        }

        void DriveSessionOutcome(
            string testRoot,
            GuidedQuestOutcomeKind? guidedKind,
            TrustedHybridOutcomeKind? hybridKind)
        {
            var repo = new PersonalizationCandidateRepository("test_learner", testRoot);
            var questToken = new PetRobotGreetingQuestToken();

            PracticeRecognitionAttempt attemptToReturn = null;
            if (guidedKind.HasValue)
            {
                var outcome = new GuidedQuestOutcome(
                    guidedKind.Value,
                    "GREETING", "GREETING",
                    true, true, true,
                    guidedKind.Value == GuidedQuestOutcomeKind.StrictSuccess,
                    guidedKind.Value != GuidedQuestOutcomeKind.TechnicalFailure &&
                    guidedKind.Value != GuidedQuestOutcomeKind.WrongWord,
                    "test outcome");
                var seq = CreateTestSequence();
                var metrics = new PracticeRecognitionMetrics
                {
                    WinningPrototypeId = "test_proto",
                    StrictAccepted = true,
                    StrictExpectedWordClassAccept = true,
                    StrictQuestSuccess = guidedKind.Value == GuidedQuestOutcomeKind.StrictSuccess,
                };
                attemptToReturn = new PracticeRecognitionAttempt(
                    Guid.NewGuid().ToString("N"),
                    "GREETING", "GREETING",
                    PersonalLiveCaptureState.Completed,
                    outcome, null,
                    guidedKind.Value == GuidedQuestOutcomeKind.TechnicalFailure ? null : seq,
                    "test_route", "test_profile", 0.5, null, null, metrics);
            }

            TrustedHybridRecognitionResult hybridResult = null;
            if (hybridKind.HasValue)
            {
                hybridResult = new TrustedHybridRecognitionResult(
                    hybridKind.Value, "GREETING", null, null, null);
            }

            var adapter = new OutcomeDrivenRecognitionAdapter { NextAttempt = attemptToReturn };
            var hybridSession = new OutcomeDrivenHybridSession { NextResult = hybridResult };

            var session = new PetRobotSignPracticeSession(
                adapter, repo, hybridSession, questToken);
            Assert.AreEqual(PetRobotPracticeState.Ready, session.State);

            Assert.IsTrue(session.StartExpression());
            Assert.AreEqual(PetRobotPracticeState.Capturing, session.State);

            Assert.IsTrue(session.FinishExpression());
            Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);

            if (hybridKind.HasValue)
            {
                Assert.AreEqual(hybridKind.Value, session.QuestAuthoritativeOutcome,
                    "quest authoritative outcome must match for " + (guidedKind?.ToString() ?? "null"));
            }
            else
            {
                Assert.AreEqual(TrustedHybridOutcomeKind.TechnicalFailure,
                    session.QuestAuthoritativeOutcome,
                    "null FinalizeCapture must result in TechnicalFailure");
            }

            if (session.CanRetry)
            {
                Assert.IsTrue(session.Retry());
                Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
            }

            session.Dispose();
        }

        // === Scene behavior fixture tests (§36-§42) ===

#if UNITY_EDITOR
        static readonly string WorldScenePath =
            "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity";

        static GameObject FindDialogueSystemRoot(Scene scene)
        {
            GameObject[] roots = scene.GetRootGameObjects();
            foreach (GameObject go in roots)
            {
                if (go.name == "InWorldDialogueSystem")
                    return go;
            }
            return null;
        }

        // §36: Scene has InWorldDialogueSystem root
        [Test]
        public void Scene_HasInWorldDialogueSystemRoot()
        {
            Assert.IsTrue(File.Exists(WorldScenePath), "World scene file must exist at " + WorldScenePath);
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject dialogueSystem = FindDialogueSystemRoot(scene);
                Assert.IsNotNull(dialogueSystem,
                    "scene must contain a root GameObject named InWorldDialogueSystem");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        // §37: InWorldDialogueSystem has exactly 1 controller and 1 camera route
        [Test]
        public void Scene_DialogueSystemComponentCounts()
        {
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject dialogueSystem = FindDialogueSystemRoot(scene);
                Assert.IsNotNull(dialogueSystem, "InWorldDialogueSystem root must exist");

                var controllers = dialogueSystem.GetComponents<InWorldGreetingNpcDialogueController>();
                Assert.AreEqual(1, controllers.Length,
                    "InWorldDialogueSystem must have exactly 1 InWorldGreetingNpcDialogueController");

                var routes = dialogueSystem.GetComponents<PetRobotCandidateCameraStartupRoute>();
                Assert.AreEqual(1, routes.Length,
                    "InWorldDialogueSystem must have exactly 1 PetRobotCandidateCameraStartupRoute");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        // §38: Controller core serialized fields wired
        [Test]
        public void Scene_ControllerCoreFieldsWired()
        {
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject dialogueSystem = FindDialogueSystemRoot(scene);
                Assert.IsNotNull(dialogueSystem, "InWorldDialogueSystem root must exist");
                var ctrl = dialogueSystem.GetComponent<InWorldGreetingNpcDialogueController>();
                Assert.IsNotNull(ctrl, "controller must exist");

                var so = new SerializedObject(ctrl);
                AssertFieldNotNull(so, "targetNpcTransform",
                    "targetNpcTransform must be wired to a scene Transform");
                AssertFieldNotNull(so, "playerTransform",
                    "playerTransform must be wired to a scene Transform");
                AssertFieldNotNull(so, "dialogueCameraAnchor",
                    "dialogueCameraAnchor must be wired to DialogueCameraAnchor child");
                AssertFieldNotNull(so, "candidateCameraRoute",
                    "candidateCameraRoute must be wired to PetRobotCandidateCameraStartupRoute");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        // §39: Controller presentation fields
        [Test]
        public void Scene_ControllerPresentationFieldsWired()
        {
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject dialogueSystem = FindDialogueSystemRoot(scene);
                Assert.IsNotNull(dialogueSystem, "InWorldDialogueSystem root must exist");
                var ctrl = dialogueSystem.GetComponent<InWorldGreetingNpcDialogueController>();
                Assert.IsNotNull(ctrl, "controller must exist");

                var so = new SerializedObject(ctrl);

                var disableArray = so.FindProperty("disableDuringDialogue");
                Assert.IsNotNull(disableArray, "disableDuringDialogue property must exist");
                Assert.GreaterOrEqual(disableArray.arraySize, 2,
                    "disableDuringDialogue must contain at least 2 Behaviours " +
                    "(PetRobotWorldPlayerController + PetRobotWorldCameraFollow)");
                for (int i = 0; i < disableArray.arraySize; i++)
                {
                    var elem = disableArray.GetArrayElementAtIndex(i);
                    Assert.IsNotNull(elem.objectReferenceValue,
                        $"disableDuringDialogue[{i}] must not be null");
                }

                AssertFieldNotNull(so, "webcamPreviewRoot",
                    "webcamPreviewRoot must be wired to a GameObject (REQUIRED for dialogue presentation)");
                AssertFieldNotNull(so, "moaDialogueRoot",
                    "moaDialogueRoot must be wired to a GameObject (REQUIRED for dialogue presentation)");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        // §40: Camera route serialized fields
        [Test]
        public void Scene_CameraRouteFieldsWired()
        {
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject dialogueSystem = FindDialogueSystemRoot(scene);
                Assert.IsNotNull(dialogueSystem, "InWorldDialogueSystem root must exist");
                var route = dialogueSystem.GetComponent<PetRobotCandidateCameraStartupRoute>();
                Assert.IsNotNull(route, "camera route must exist");

                var so = new SerializedObject(route);
                AssertFieldNotNull(so, "cameraDisplay",
                    "cameraDisplay must be wired to a RawImage for webcam preview");
                AssertFieldNotNull(so, "captureConfig",
                    "captureConfig must be wired to DefaultCaptureConfig.asset");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        // §41: GREETING trigger relay exists, wired, Collider with isTrigger
        [Test]
        public void Scene_HasGreetingTriggerRelayWithCollider()
        {
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject[] roots = scene.GetRootGameObjects();
                GreetingNpcTriggerRelay relay = null;
                foreach (GameObject rootGo in roots)
                {
                    relay = rootGo.GetComponentInChildren<GreetingNpcTriggerRelay>(true);
                    if (relay != null) break;
                }
                Assert.IsNotNull(relay,
                    "scene must have a GreetingNpcTriggerRelay on a greeting NPC child");

                var so = new SerializedObject(relay);
                AssertFieldNotNull(so, "dialogueController",
                    "GreetingNpcTriggerRelay.dialogueController must be wired to " +
                    "InWorldGreetingNpcDialogueController on InWorldDialogueSystem");

                Collider col = relay.GetComponent<Collider>();
                Assert.IsNotNull(col,
                    "GreetingNpcTriggerRelay GameObject must have a Collider component");
                Assert.IsTrue(col.isTrigger,
                    "Collider on GreetingNpcTriggerRelay must have isTrigger = true");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        // §42: Greeting and education NPC are different Transforms
        [Test]
        public void Scene_GreetingAndEducationNpcAreDifferent()
        {
            var scene = EditorSceneManager.OpenScene(WorldScenePath, OpenSceneMode.Additive);
            try
            {
                GameObject dialogueSystem = FindDialogueSystemRoot(scene);
                Assert.IsNotNull(dialogueSystem, "InWorldDialogueSystem root must exist");
                var ctrl = dialogueSystem.GetComponent<InWorldGreetingNpcDialogueController>();
                Assert.IsNotNull(ctrl, "InWorldGreetingNpcDialogueController must exist");
                var ctrlSo = new SerializedObject(ctrl);
                var greetingNpcProp = ctrlSo.FindProperty("targetNpcTransform");
                Assert.IsNotNull(greetingNpcProp, "targetNpcTransform property must exist");
                var greetingNpcObj = greetingNpcProp.objectReferenceValue as Transform;
                Assert.IsNotNull(greetingNpcObj,
                    "targetNpcTransform must be wired (greeting NPC target)");

                GameObject[] roots = scene.GetRootGameObjects();
                PetRobotPortfolioQuestController questCtrl = null;
                foreach (GameObject rootGo in roots)
                {
                    questCtrl = rootGo.GetComponentInChildren<PetRobotPortfolioQuestController>(true);
                    if (questCtrl != null) break;
                }
                Assert.IsNotNull(questCtrl,
                    "PetRobotPortfolioQuestController must exist in scene");
                var questSo = new SerializedObject(questCtrl);
                var educationNpcProp = questSo.FindProperty("educationNpcTransform");
                Assert.IsNotNull(educationNpcProp, "educationNpcTransform property must exist");
                var educationNpcObj = educationNpcProp.objectReferenceValue as Transform;
                Assert.IsNotNull(educationNpcObj,
                    "educationNpcTransform must be wired (portfolio quest provider NPC)");

                Assert.AreNotEqual(greetingNpcObj.GetInstanceID(), educationNpcObj.GetInstanceID(),
                    "greeting targetNpcTransform and portfolio educationNpcTransform " +
                    "must be DIFFERENT Transforms");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }

        static void AssertFieldNotNull(SerializedObject so, string fieldName, string message)
        {
            var prop = so.FindProperty(fieldName);
            Assert.IsNotNull(prop, $"serialized property '{fieldName}' must exist on component");
            Assert.IsNotNull(prop.objectReferenceValue, message);
        }
#endif

        // --- helpers ---

        void BeginRecognition()
        {
            if (controller.State == InWorldDialogueState.QuestPrompt)
            {
                testTime += controller.QuestPromptSeconds + 0.01f;
                controller.DriveUpdate();
            }
        }

        string SimulateOutcome(TrustedHybridOutcomeKind kind)
        {
            string id = Guid.NewGuid().ToString("N");
            controller.InjectOutcomeForTest(id, kind);
            controller.DriveUpdate();
            testTime += controller.AnalysisDisplaySeconds + 0.01f;
            controller.DriveUpdate();
            return id;
        }

        static readonly string FakeSha256 = new string('a', 64);

        static RecognitionSequence CreateTestSequence()
        {
            int length = 10;
            int landmarks = RecognitionSequence.RequiredLandmarkCount;
            var coords = new double[length * landmarks * 3];
            var mask = new bool[length * landmarks];
            for (int i = 0; i < mask.Length; i++) mask[i] = true;
            return new RecognitionSequence(
                "test_sample", "GREETING", "normal", "train",
                FakeSha256, FakeSha256, length, landmarks, coords, mask);
        }

        static string ComputeDirectorySnapshot(string rootDir)
        {
            var sb = new StringBuilder();
            var files = Directory.GetFiles(rootDir, "*", SearchOption.AllDirectories);
            Array.Sort(files, StringComparer.Ordinal);
            using (var sha = SHA256.Create())
            {
                foreach (string file in files)
                {
                    string rel = file.Substring(rootDir.Length).Replace('\\', '/');
                    long size = new FileInfo(file).Length;
                    byte[] hash = sha.ComputeHash(File.ReadAllBytes(file));
                    sb.Append(rel).Append('|').Append(size).Append('|');
                    foreach (byte b in hash) sb.Append(b.ToString("x2"));
                    sb.AppendLine();
                }
            }
            return sb.ToString();
        }

        // R4 relay helpers.
        //
        // Unity 6.21 EditMode refuses SendMessage on a component whose behaviour is not
        // "running", so these pure unit tests invoke the relay's own private handlers directly
        // and assert that each handler exists. Nothing about the relay or the controller is
        // faked, and no Unity error is suppressed. The ENGINE side - whether physics actually
        // dispatches OnTriggerEnter/Exit and whether the disable lifecycle fires - is not
        // covered here and remains runtime/human coverage.
        static void InvokeRelayHandler(GreetingNpcTriggerRelay relay, string method,
                                       params object[] arguments)
        {
            System.Reflection.MethodInfo info = typeof(GreetingNpcTriggerRelay).GetMethod(
                method,
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Public);
            Assert.IsNotNull(info, "GreetingNpcTriggerRelay must declare " + method);
            info.Invoke(relay, arguments);
        }

        static void RelayEnter(GreetingNpcTriggerRelay relay, Collider collider)
        {
            InvokeRelayHandler(relay, "OnTriggerEnter", collider);
        }

        static void RelayExit(GreetingNpcTriggerRelay relay, Collider collider)
        {
            InvokeRelayHandler(relay, "OnTriggerExit", collider);
        }

        static void RelayDisable(GreetingNpcTriggerRelay relay)
        {
            InvokeRelayHandler(relay, "OnDisable");
        }

        GameObject CreateRelayFixture(out GreetingNpcTriggerRelay relay, out InWorldGreetingNpcDialogueController ctrl)
        {
            var go = new GameObject("RelayTest");
            ctrl = go.AddComponent<InWorldGreetingNpcDialogueController>();
            var g = new FakeGateVerifier { Ready = true };
            var r = new FakeRuntimeOwner();
            ctrl.ConfigureForTests(npc, player, 5f, g,
                new PetRobotLearnerRootIdentity("test_learner"),
                r, new FakeCameraController(), new FakePlayerLock(), new FakeNpcPresenter());
            relay = go.AddComponent<GreetingNpcTriggerRelay>();
#if UNITY_EDITOR
            var so = new UnityEditor.SerializedObject(relay);
            so.FindProperty("dialogueController").objectReferenceValue = ctrl;
            so.ApplyModifiedPropertiesWithoutUndo();
#endif
            return go;
        }

        void DestroyRelayFixture(GameObject go)
        {
            if (go != null) UnityEngine.Object.DestroyImmediate(go);
        }

        Collider CreateTaggedCollider(string tag)
        {
            var go = new GameObject("Col_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            go.tag = tag;
            return go.AddComponent<BoxCollider>();
        }

        // --- test doubles ---

        sealed class FakeGateVerifier : IQuestGateVerifier
        {
            public bool Ready;
            public string Failure = string.Empty;
            public bool ThrowOnVerify;

            public string LastFailure => Failure;

            public bool VerifyOverallEnrollmentGateReady()
            {
                if (ThrowOnVerify) throw new InvalidOperationException("gate exception");
                return Ready;
            }
        }

        sealed class FakeRuntimeOwner : IInWorldRecognitionRuntimeOwner
        {
            public long Generation = 1;
            public bool Disposed;
            public bool RequestAttemptResult = true;
            public bool RetrySessionResult = true;
            public bool ThrowOnRequestAttempt;
            public bool ThrowOnRetrySession;
            public int RequestAttemptCount;
            public int RetrySessionCount;
            public int CandidateWriteCount;
            public string FailureMessage;
            public bool FirstObservationReady;

            public bool IsActive => !Disposed;
            public long FrozenGeneration => Generation;
            public PetRobotSignPracticeSession Session => null;
            public PetRobotNeutralAutoCaptureCoordinator Coordinator => null;
            public string InitializationFailure => FailureMessage;

            public bool RequestAttempt()
            {
                if (ThrowOnRequestAttempt)
                    throw new InvalidOperationException("request attempt exception");
                RequestAttemptCount++;
                return RequestAttemptResult;
            }

            public bool RetrySession()
            {
                if (ThrowOnRetrySession)
                    throw new InvalidOperationException("retry session exception");
                RetrySessionCount++;
                return RetrySessionResult;
            }

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

        sealed class OutcomeDrivenRecognitionAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            public PracticeRecognitionAttempt NextAttempt;
            public PersonalLiveCaptureState CaptureState => PersonalLiveCaptureState.Ready;
            public string CaptureSessionId => "scripted_session_" + Guid.NewGuid().ToString("N");
            public string ActiveProfileId => "test_profile";
            public bool CandidateRouteActive => true;
            public bool SetExpectedWord(string word) => true;
            public bool BeginCapture() => true;
            public PracticeRecognitionAttempt FinalizeCapture() => NextAttempt;
            public bool ResetCapture() => true;
        }

        sealed class OutcomeDrivenHybridSession : IQuestHybridRecognitionSession
        {
            public TrustedHybridRecognitionResult NextResult;
            public long Generation => 1;
            public TrustedHybridRecognitionResult Recognize(
                string querySessionId, RecognitionSequence query, string expectedWord) => NextResult;
        }
    }

    // === Step 10M v2.4a — Behavior Tests for Guided Quest Flow ===
    [TestFixture]
    public sealed class InWorldGreetingGuidedQuestFlowV24aTests
    {
        GameObject root;
        InWorldGreetingNpcDialogueController controller;
        Transform npc;
        Transform player;
        FakeGateVerifier gate;
        FakeRuntimeOwner runtime;
        FakeCameraController camera;
        FakePlayerLock playerLock;
        FakeNpcPresenter npcPresenter;
        float testTime;

        [SetUp]
        public void SetUp()
        {
            testTime = 0f;
            root = new GameObject("TestRoot_v24a");
            controller = root.AddComponent<InWorldGreetingNpcDialogueController>();
            npc = new GameObject("NPC_v24a").transform;
            player = new GameObject("Player_v24a").transform;
            npc.position = Vector3.zero;
            player.position = Vector3.forward * 2f;
            gate = new FakeGateVerifier();
            runtime = new FakeRuntimeOwner();
            camera = new FakeCameraController();
            playerLock = new FakePlayerLock();
            npcPresenter = new FakeNpcPresenter();

            controller.ConfigureForTests(
                npc, player, 5f, gate,
                new PetRobotLearnerRootIdentity("test_learner"),
                runtime, camera, playerLock, npcPresenter);
            controller.SetTimeSourceForTest(() => testTime);
            controller.SetPlayerInsideGreetingTrigger(true);
        }

        [TearDown]
        public void TearDown()
        {
            if (root != null) UnityEngine.Object.DestroyImmediate(root);
            if (npc != null) UnityEngine.Object.DestroyImmediate(npc.gameObject);
            if (player != null) UnityEngine.Object.DestroyImmediate(player.gameObject);
        }

        // B-01: E key → QuestPrompt (not TrackingPreparation)
        [Test]
        public void TryStartDialogue_SetsQuestPrompt_NotTrackingPreparation()
        {
            gate.Ready = true;
            Assert.IsTrue(controller.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, controller.State);
            Assert.AreNotEqual(InWorldDialogueState.TrackingPreparation, controller.State);
        }

        // B-02: Quest prompt text contains Korean quest instruction
        [Test]
        public void QuestPrompt_HasKoreanQuestText()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            Assert.IsFalse(string.IsNullOrEmpty(controller.LastFeedbackText),
                "quest prompt must set feedback text");
            Assert.IsTrue(controller.LastFeedbackText.Contains("인사"),
                "quest prompt text must contain '인사' (greeting)");
        }

        // B-03: Prompt timer auto-transitions to TrackingPreparation after questPromptSeconds
        [Test]
        public void PromptTimer_AutoTransitions_AfterQuestPromptSeconds()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, controller.State);
            Assert.IsFalse(controller.RecognitionInitiated);

            testTime += controller.QuestPromptSeconds + 0.01f;
            controller.DriveUpdate();

            Assert.IsTrue(controller.RecognitionInitiated);
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
        }

        // B-04: Prompt timer does NOT fire before questPromptSeconds
        [Test]
        public void PromptTimer_DoesNotFire_BeforeQuestPromptSeconds()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, controller.State);

            testTime += controller.QuestPromptSeconds * 0.5f;
            controller.DriveUpdate();

            Assert.IsFalse(controller.RecognitionInitiated);
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, controller.State);
        }

        // B-05: Duplicate guard — second DriveUpdate after timer doesn't re-trigger
        [Test]
        public void PromptTimer_DuplicateGuard_NoSecondTrigger()
        {
            gate.Ready = true;
            controller.TryStartDialogue();

            testTime += controller.QuestPromptSeconds + 0.01f;
            controller.DriveUpdate();
            Assert.AreEqual(1, runtime.RequestAttemptCount);

            testTime += 1f;
            controller.DriveUpdate();
            Assert.AreEqual(1, runtime.RequestAttemptCount,
                "second DriveUpdate must not call RequestAttempt again");
        }

        // B-06: Tracking timeout → TechnicalFailure
        [Test]
        public void TrackingTimeout_TransitionsToTechnicalFailure()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);

            testTime += controller.TrackingPreparationTimeoutSeconds + 0.01f;
            controller.DriveUpdate();

            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsFalse(string.IsNullOrEmpty(controller.LastFeedbackText),
                "timeout must set Korean feedback text");
        }

        // B-07: FirstObservation received prevents timeout
        [Test]
        public void FirstObservation_PreventsTimeout()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();

            runtime.FirstObservationReady = true;
            testTime += 1f;
            controller.DriveUpdate();
            Assert.IsTrue(controller.FirstObservationReceived);

            testTime += controller.TrackingPreparationTimeoutSeconds + 1f;
            controller.DriveUpdate();
            Assert.AreNotEqual(InWorldDialogueState.Feedback, controller.State,
                "must not timeout after first observation received");
        }

        // B-08: Null integrator factory → InitializationFailure → Feedback
        [Test]
        public void NullIntegrator_TypedInitFailure()
        {
            var failRuntime = new FakeRuntimeOwner();
            failRuntime.FailureMessage = "providerIntegrator is not assigned in Inspector";
            controller.SetRuntimeFactoryForTest(() => failRuntime);

            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();

            Assert.AreEqual(InWorldDialogueState.Feedback, controller.State);
            Assert.AreEqual(InWorldDialogueOutcome.TechnicalFailure, controller.LastOutcome);
            Assert.IsTrue(failRuntime.Disposed);
        }

        // B-09: Analyzing hold — outcome queued, held for analysisDisplaySeconds
        [Test]
        public void AnalyzingHold_OutcomeNotAppliedBeforeDuration()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();

            string id = Guid.NewGuid().ToString("N");
            controller.InjectOutcomeForTest(id, TrustedHybridOutcomeKind.StrictSuccess);
            controller.DriveUpdate();

            Assert.AreEqual(InWorldDialogueState.Analyzing, controller.State,
                "must enter Analyzing immediately after outcome queued");
            Assert.IsFalse(controller.ItemGranted,
                "outcome must not be applied during analyzing hold");
        }

        // B-10: Analyzing hold — outcome applied after analysisDisplaySeconds
        [Test]
        public void AnalyzingHold_OutcomeAppliedAfterDuration()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();

            string id = Guid.NewGuid().ToString("N");
            controller.InjectOutcomeForTest(id, TrustedHybridOutcomeKind.StrictSuccess);
            controller.DriveUpdate();
            Assert.AreEqual(InWorldDialogueState.Analyzing, controller.State);

            testTime += controller.AnalysisDisplaySeconds + 0.01f;
            controller.DriveUpdate();

            Assert.AreEqual(InWorldDialogueState.QuestComplete, controller.State);
            Assert.IsTrue(controller.ItemGranted);
        }

        // B-11: CloseDialogue clears v2.4a state
        [Test]
        public void CloseDialogue_ClearsV24aState()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            Assert.IsTrue(controller.RecognitionInitiated);

            controller.CloseDialogue();
            Assert.IsFalse(controller.RecognitionInitiated,
                "recognitionInitiated must be cleared on close");
            Assert.IsFalse(controller.FirstObservationReceived,
                "firstObservationReceived must be cleared on close");
        }

        // B-12: Retry clears v2.4a tracking state
        [Test]
        public void Retry_ClearsTrackingState()
        {
            gate.Ready = true;
            controller.TryStartDialogue();
            BeginRecognition();
            runtime.FirstObservationReady = true;
            testTime += 1f;
            controller.DriveUpdate();
            Assert.IsTrue(controller.FirstObservationReceived);

            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);

            Assert.IsTrue(controller.TryRetry());
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, controller.State);
        }

        // B-13: Source — no FindObjectOfType or FindFirstObjectByType
        [Test]
        public void Source_NoFindObjectOfTypeInController()
        {
            string source = System.IO.File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsFalse(source.Contains("FindObjectOfType"),
                "controller must not use FindObjectOfType");
            Assert.IsFalse(source.Contains("FindFirstObjectByType"),
                "controller must not use FindFirstObjectByType");
        }

        // B-14: InWorldDialogueState enum has QuestPrompt
        [Test]
        public void DialogueState_HasQuestPrompt()
        {
            Assert.IsTrue(Enum.IsDefined(typeof(InWorldDialogueState),
                InWorldDialogueState.QuestPrompt));
        }

        // B-15: InWorldDialogueState enum has all guided sub-states
        [Test]
        public void DialogueState_HasAllGuidedSubStates()
        {
            Assert.IsTrue(Enum.IsDefined(typeof(InWorldDialogueState),
                InWorldDialogueState.TrackingPreparation));
            Assert.IsTrue(Enum.IsDefined(typeof(InWorldDialogueState),
                InWorldDialogueState.NeutralPreparation));
            Assert.IsTrue(Enum.IsDefined(typeof(InWorldDialogueState),
                InWorldDialogueState.ReadyForGreeting));
            Assert.IsTrue(Enum.IsDefined(typeof(InWorldDialogueState),
                InWorldDialogueState.Capturing));
            Assert.IsTrue(Enum.IsDefined(typeof(InWorldDialogueState),
                InWorldDialogueState.Analyzing));
        }

        // B-16: FeedbackPresenter handles QuestPrompt case
        [Test]
        public void FeedbackPresenter_HandlesQuestPromptState()
        {
            string source = System.IO.File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueFeedbackPresenter.cs");
            Assert.IsTrue(source.Contains("QuestPrompt"),
                "presenter must handle QuestPrompt state");
        }

        // B-17: Controller has providerIntegrator property
        [Test]
        public void Controller_HasProviderIntegratorProperty()
        {
            Assert.IsNull(controller.ProviderIntegrator,
                "ProviderIntegrator should be null in test fixture (no scene wiring)");
        }

        // B-18: BuildRecognitionRuntime passes providerIntegrator to production owner
        [Test]
        public void Source_BuildRecognitionRuntime_PassesProviderIntegrator()
        {
            string source = System.IO.File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsTrue(source.Contains("providerIntegrator)"),
                "BuildRecognitionRuntime must pass providerIntegrator");
        }

        // B-19: ProductionRuntimeOwner calls BindForDiagnostic
        [Test]
        public void Source_ProductionRuntimeOwner_CallsBindForDiagnostic()
        {
            string source = System.IO.File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsTrue(source.Contains("BindForDiagnostic(explicitIntegrator)"),
                "must call BindForDiagnostic with explicit integrator");
        }

        // B-20: ProductionRuntimeOwner null integrator check at top
        [Test]
        public void Source_ProductionRuntimeOwner_NullIntegratorCheck()
        {
            string source = System.IO.File.ReadAllText(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/InWorldGreetingNpcDialogueController.cs");
            Assert.IsTrue(source.Contains("explicitIntegrator == null"),
                "must check for null explicitIntegrator");
            Assert.IsTrue(source.Contains("providerIntegrator is not assigned"),
                "must set typed failure message for null integrator");
        }

#if UNITY_EDITOR
        // B-21: Scene providerIntegrator field exists
        [Test]
        public void Scene_ProviderIntegratorFieldExists()
        {
            string scenePath = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity";
            if (!System.IO.File.Exists(scenePath))
            {
                Assert.Inconclusive("World scene not found at " + scenePath);
                return;
            }
            var scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Additive);
            try
            {
                GameObject[] roots = scene.GetRootGameObjects();
                InWorldGreetingNpcDialogueController ctrl = null;
                foreach (GameObject go in roots)
                {
                    ctrl = go.GetComponentInChildren<InWorldGreetingNpcDialogueController>(true);
                    if (ctrl != null) break;
                }
                if (ctrl == null)
                {
                    Assert.Inconclusive("InWorldGreetingNpcDialogueController not found in scene");
                    return;
                }
                var so = new SerializedObject(ctrl);
                var prop = so.FindProperty("providerIntegrator");
                Assert.IsNotNull(prop, "providerIntegrator serialized property must exist");
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
            }
        }
#endif

        // --- helpers ---

        void BeginRecognition()
        {
            if (controller.State == InWorldDialogueState.QuestPrompt)
            {
                testTime += controller.QuestPromptSeconds + 0.01f;
                controller.DriveUpdate();
            }
        }

        string SimulateOutcome(TrustedHybridOutcomeKind kind)
        {
            string id = Guid.NewGuid().ToString("N");
            controller.InjectOutcomeForTest(id, kind);
            controller.DriveUpdate();
            testTime += controller.AnalysisDisplaySeconds + 0.01f;
            controller.DriveUpdate();
            return id;
        }

        // --- test doubles ---

        sealed class FakeGateVerifier : IQuestGateVerifier
        {
            public bool Ready;
            public string Failure = string.Empty;
            public string LastFailure => Failure;
            public bool VerifyOverallEnrollmentGateReady() => Ready;
        }

        sealed class FakeRuntimeOwner : IInWorldRecognitionRuntimeOwner
        {
            public bool Disposed;
            public bool RequestAttemptResult = true;
            public bool RetrySessionResult = true;
            public int RequestAttemptCount;
            public bool FirstObservationReady;
            public string FailureMessage;
            public bool IsActive => !Disposed;
            public long FrozenGeneration => 1;
            public PetRobotSignPracticeSession Session => null;
            public PetRobotNeutralAutoCaptureCoordinator Coordinator => null;
            public string InitializationFailure => FailureMessage;
            public bool RequestAttempt() { RequestAttemptCount++; return RequestAttemptResult; }
            public bool RetrySession() => RetrySessionResult;

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
            public bool Restored;
            public void SaveState() { Restored = false; }
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
            public void TriggerGreeting() { }
            public void TriggerConfused() { }
            public void TriggerGiveItem() { }
        }
    }
}
