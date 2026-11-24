using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class GreetingQuestModePracticeReuseV1Tests
    {
        // ============================================================
        // §1  Handoff Token (behavior)
        // ============================================================

        [Test]
        public void QuestToken_GeneratesNonEmptyId()
        {
            var token = new PetRobotGreetingQuestToken();
            Assert.IsFalse(string.IsNullOrWhiteSpace(token.Id));
            Assert.AreEqual(32, token.Id.Length);
        }

        [Test]
        public void QuestToken_InternalConstructor_AcceptsFixedId()
        {
            var token = new PetRobotGreetingQuestToken("abc123");
            Assert.AreEqual("abc123", token.Id);
        }

        [Test]
        public void QuestToken_InternalConstructor_RejectsEmpty()
        {
            Assert.Throws<ArgumentException>(() => new PetRobotGreetingQuestToken(""));
            Assert.Throws<ArgumentException>(() => new PetRobotGreetingQuestToken("  "));
        }

        [Test]
        public void QuestResult_IsSuccess_TrueForStrictAndLowConfidence()
        {
            var strict = new PetRobotGreetingQuestResult("t1",
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
            var low = new PetRobotGreetingQuestResult("t2",
                TrustedHybridOutcomeKind.LowConfidenceCorrect, "ok", "", "");
            var wrong = new PetRobotGreetingQuestResult("t3",
                TrustedHybridOutcomeKind.WrongWord, "wrong", "", "");
            var tech = new PetRobotGreetingQuestResult("t4",
                TrustedHybridOutcomeKind.TechnicalFailure, "fail", "Err", "msg");
            Assert.IsTrue(strict.IsSuccess);
            Assert.IsTrue(low.IsSuccess);
            Assert.IsFalse(wrong.IsSuccess);
            Assert.IsFalse(tech.IsSuccess);
        }

        // ============================================================
        // §2  Channel Quest Handoff — identity required (behavior, coverage #2 #4)
        // ============================================================

        [Test]
        public void Channel_BeginGreetingQuest_SetsQuestModeWithIdentity()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            var prior = Mastered(9);
            var identity = DefaultIdentity();
            Assert.IsTrue(channel.BeginGreetingQuest("World", "Practice",
                Vector3.zero, token, prior, identity));
            Assert.AreEqual(PetRobotHandoffMode.GreetingQuest, channel.PendingMode);
            Assert.AreSame(token, channel.PendingQuestToken);
            Assert.AreSame(prior, channel.PendingPriorCompletion);
            Assert.AreSame(identity, channel.PendingLearnerIdentity);
        }

        [Test]
        public void Channel_BeginGreetingQuest_RejectsNullIdentity()
        {
            var channel = new PetRobotWorldHandoffChannel();
            Assert.IsFalse(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, new PetRobotGreetingQuestToken(), Mastered(9), null));
        }

        [Test]
        public void Channel_BeginGreetingQuest_RejectsBlankLearnerId()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var identity = new PetRobotLearnerRootIdentity("  ");
            Assert.IsFalse(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, new PetRobotGreetingQuestToken(), Mastered(9), identity));
        }

        [Test]
        public void Channel_BeginGreetingQuest_RejectsNullToken()
        {
            var channel = new PetRobotWorldHandoffChannel();
            Assert.IsFalse(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, null, Mastered(9), DefaultIdentity()));
        }

        [Test]
        public void Channel_BeginGreetingQuest_RejectsNullPrior()
        {
            var channel = new PetRobotWorldHandoffChannel();
            Assert.IsFalse(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, new PetRobotGreetingQuestToken(), null, DefaultIdentity()));
        }

        [Test]
        public void Channel_BeginGreetingQuest_RejectsDuplicate()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            Assert.IsTrue(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, token, Mastered(9), DefaultIdentity()));
            Assert.IsFalse(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, new PetRobotGreetingQuestToken(), Mastered(9), DefaultIdentity()));
        }

        [Test]
        public void Channel_PublishQuestResult_OnlyAcceptsMatchingToken()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, Mastered(9), DefaultIdentity());
            var mismatch = new PetRobotGreetingQuestResult("wrong_id",
                TrustedHybridOutcomeKind.StrictSuccess, "", "", "");
            Assert.IsFalse(channel.PublishQuestResult(mismatch));
            var match = new PetRobotGreetingQuestResult(token.Id,
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
            Assert.IsTrue(channel.PublishQuestResult(match));
        }

        [Test]
        public void Channel_QuestResult_SurvivesTryConsumeCompleted()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            var prior = Mastered(9);
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, prior, DefaultIdentity());
            var result = new PetRobotGreetingQuestResult(token.Id,
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
            Assert.IsTrue(channel.PublishQuestResult(result));
            Assert.IsTrue(channel.PublishCompleted(prior));
            PetRobotWorldReturnPayload payload;
            Assert.IsTrue(channel.TryConsumeCompleted("W", out payload));
            PetRobotGreetingQuestResult consumed;
            Assert.IsTrue(channel.TryConsumeQuestResult(out consumed));
            Assert.AreSame(result, consumed);
        }

        [Test]
        public void Channel_TryConsumeQuestResult_ConsumeOnlyOnce()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, Mastered(9), DefaultIdentity());
            channel.PublishQuestResult(new PetRobotGreetingQuestResult(
                token.Id, TrustedHybridOutcomeKind.StrictSuccess, "ok", "", ""));
            PetRobotGreetingQuestResult r1;
            Assert.IsTrue(channel.TryConsumeQuestResult(out r1));
            PetRobotGreetingQuestResult r2;
            Assert.IsFalse(channel.TryConsumeQuestResult(out r2));
        }

        [Test]
        public void Channel_NormalMode_HasNoQuestToken()
        {
            var channel = new PetRobotWorldHandoffChannel();
            channel.Begin("W", "P", Vector3.zero);
            Assert.AreEqual(PetRobotHandoffMode.Normal, channel.PendingMode);
            Assert.IsNull(channel.PendingQuestToken);
        }

        [Test]
        public void Channel_HasQuestResult_FalseBeforePublish()
        {
            var channel = new PetRobotWorldHandoffChannel();
            Assert.IsFalse(channel.HasQuestResult);
        }

        [Test]
        public void Channel_HasQuestResult_TrueAfterPublish()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, Mastered(9), DefaultIdentity());
            channel.PublishQuestResult(new PetRobotGreetingQuestResult(
                token.Id, TrustedHybridOutcomeKind.StrictSuccess, "ok", "", ""));
            Assert.IsTrue(channel.HasQuestResult);
        }

        // ============================================================
        // §3  CommitQuestReturn (behavior, coverage #14 #15)
        // ============================================================

        [Test]
        public void CommitQuestReturn_PublishesBothAtomically()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            var prior = Mastered(9);
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, prior, DefaultIdentity());
            var result = new PetRobotGreetingQuestResult(token.Id,
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
            Assert.IsTrue(channel.CommitQuestReturn(result, prior));
            Assert.IsTrue(channel.HasQuestResult);
            Assert.IsTrue(channel.HasPublishedCompletion);
        }

        [Test]
        public void CommitQuestReturn_RejectsMismatchedToken()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, Mastered(9), DefaultIdentity());
            var badResult = new PetRobotGreetingQuestResult("wrong",
                TrustedHybridOutcomeKind.StrictSuccess, "", "", "");
            Assert.IsFalse(channel.CommitQuestReturn(badResult, Mastered(9)));
            Assert.IsFalse(channel.HasQuestResult);
        }

        [Test]
        public void CommitQuestReturn_RejectsNullResult()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, Mastered(9), DefaultIdentity());
            Assert.IsFalse(channel.CommitQuestReturn(null, Mastered(9)));
        }

        [Test]
        public void CommitQuestReturn_RejectsNullCompletion()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, Mastered(9), DefaultIdentity());
            var result = new PetRobotGreetingQuestResult(token.Id,
                TrustedHybridOutcomeKind.StrictSuccess, "", "", "");
            Assert.IsFalse(channel.CommitQuestReturn(result, null));
        }

        [Test]
        public void CommitQuestReturn_RejectsNormalMode()
        {
            var channel = new PetRobotWorldHandoffChannel();
            channel.Begin("W", "P", Vector3.zero);
            var result = new PetRobotGreetingQuestResult("any",
                TrustedHybridOutcomeKind.StrictSuccess, "", "", "");
            Assert.IsFalse(channel.CommitQuestReturn(result, Mastered(9)));
        }

        [Test]
        public void CommitQuestReturn_RejectsDoublePending()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            var prior = Mastered(9);
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, prior, DefaultIdentity());
            var result = new PetRobotGreetingQuestResult(token.Id,
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
            Assert.IsTrue(channel.CommitQuestReturn(result, prior));
            Assert.IsFalse(channel.CommitQuestReturn(result, prior));
        }

        [Test]
        public void CommitQuestReturn_ResultConsumableAfterCompletionConsumed()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            var prior = Mastered(9);
            channel.BeginGreetingQuest("W", "P", Vector3.zero, token, prior, DefaultIdentity());
            var result = new PetRobotGreetingQuestResult(token.Id,
                TrustedHybridOutcomeKind.LowConfidenceCorrect, "lcc", "", "");
            Assert.IsTrue(channel.CommitQuestReturn(result, prior));
            PetRobotWorldReturnPayload payload;
            Assert.IsTrue(channel.TryConsumeCompleted("W", out payload));
            Assert.AreSame(prior, payload.Completion);
            PetRobotGreetingQuestResult consumed;
            Assert.IsTrue(channel.TryConsumeQuestResult(out consumed));
            Assert.AreEqual("lcc", consumed.FeedbackText);
        }

        // ============================================================
        // §4  Flow Quest Restore + RollbackToDialogueOpen (behavior)
        // ============================================================

        [Test]
        public void Flow_RestoreAwaitingGreetingAttempt_SetsCorrectState()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsTrue(flow.RestoreAwaitingGreetingAttempt(Mastered(9)));
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingGreetingAttempt, flow.State);
            Assert.IsNotNull(flow.Snapshot);
        }

        [Test]
        public void Flow_RestoreAwaitingGreetingAttempt_RejectsNull()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsFalse(flow.RestoreAwaitingGreetingAttempt(null));
        }

        [Test]
        public void Flow_RestoreAwaitingGreetingAttempt_RejectsWhenAlreadyAdvanced()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            flow.ObserveCompleted(Mastered(9));
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            Assert.IsFalse(flow.RestoreAwaitingGreetingAttempt(Mastered(9)));
        }

        [Test]
        public void Flow_ObserveQuestResult_StoresResult()
        {
            var flow = Observed(Mastered(9));
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            flow.TryCompleteDialogue(true);
            var result = new PetRobotGreetingQuestResult("t1",
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
            Assert.IsTrue(flow.ObserveQuestResult(result));
            Assert.AreSame(result, flow.LastQuestResult);
            Assert.AreEqual("ok", flow.QuestFeedbackText);
        }

        [Test]
        public void Flow_ObserveQuestResult_RejectsDuplicate()
        {
            var flow = Observed(Mastered(9));
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            flow.TryCompleteDialogue(true);
            Assert.IsTrue(flow.ObserveQuestResult(new PetRobotGreetingQuestResult("t1",
                TrustedHybridOutcomeKind.StrictSuccess, "first", "", "")));
            Assert.IsFalse(flow.ObserveQuestResult(new PetRobotGreetingQuestResult("t2",
                TrustedHybridOutcomeKind.StrictSuccess, "second", "", "")));
            Assert.AreEqual("first", flow.QuestFeedbackText);
        }

        [Test]
        public void Flow_ObserveQuestResult_RejectedOutsideGreetingAttempt()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsFalse(flow.ObserveQuestResult(new PetRobotGreetingQuestResult("t1",
                TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "")));
        }

        [Test]
        public void Flow_CompleteGreetingQuest_AdvancesToDemoCompleted()
        {
            var flow = Observed(Mastered(9));
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            flow.TryCompleteDialogue(true);
            Assert.IsTrue(flow.TryCompleteGreetingQuest());
            Assert.AreEqual(PetRobotPortfolioQuestState.DemoCompleted, flow.State);
        }

        [Test]
        public void Flow_CompleteGreetingQuest_RejectsFromWrongState()
        {
            var flow = Observed(Mastered(9));
            Assert.IsFalse(flow.TryCompleteGreetingQuest());
        }

        [Test]
        public void Flow_RollbackToDialogueOpen_FromAwaitingGreeting()
        {
            var flow = Observed(Mastered(9));
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            flow.TryCompleteDialogue(true);
            Assert.IsTrue(flow.RollbackToDialogueOpen());
            Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, flow.State);
            Assert.IsNull(flow.LastQuestResult);
        }

        [Test]
        public void Flow_RollbackToDialogueOpen_RejectsFromWrongState()
        {
            var flow = Observed(Mastered(9));
            Assert.IsFalse(flow.RollbackToDialogueOpen());
        }

        [Test]
        public void Flow_RollbackToDialogueOpen_AllowsRetry()
        {
            var flow = Observed(Mastered(9));
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            flow.TryCompleteDialogue(true);
            Assert.IsTrue(flow.RollbackToDialogueOpen());
            Assert.IsTrue(flow.TryCompleteDialogue(true));
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingGreetingAttempt, flow.State);
        }

        // ============================================================
        // §5  Production Verifier binding (behavior, coverage #1 #2)
        // ============================================================

        [Test]
        public void ProductionQuestGateVerifier_RequiresNonNullIdentity()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new ProductionQuestGateVerifier(null));
        }

        [Test]
        public void ProductionQuestGateVerifier_ConstructsWithIdentity()
        {
            var identity = new PetRobotLearnerRootIdentity("test_learner", "/test/root");
            var verifier = new ProductionQuestGateVerifier(identity);
            Assert.IsNotNull(verifier);
            Assert.IsNotNull(verifier as IQuestGateVerifier);
        }

        [Test]
        public void LearnerRootIdentity_DefaultLearnerId()
        {
            var identity = new PetRobotLearnerRootIdentity(null);
            Assert.AreEqual("local_learner", identity.LearnerId);
            Assert.IsFalse(identity.HasExplicitRoot);
        }

        [Test]
        public void LearnerRootIdentity_ExplicitRoot()
        {
            var identity = new PetRobotLearnerRootIdentity("user1", "/custom/root");
            Assert.AreEqual("user1", identity.LearnerId);
            Assert.AreEqual("/custom/root", identity.ExplicitRoot);
            Assert.IsTrue(identity.HasExplicitRoot);
        }

        [Test]
        public void Channel_BeginGreetingQuest_CarriesSameIdentityObject()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var token = new PetRobotGreetingQuestToken();
            var prior = Mastered(9);
            var identity = new PetRobotLearnerRootIdentity("test_learner", "/same/root");
            Assert.IsTrue(channel.BeginGreetingQuest("W", "P",
                Vector3.zero, token, prior, identity));
            Assert.AreSame(identity, channel.PendingLearnerIdentity,
                "same identity object must be carried for same-root guarantee");
        }

        // ============================================================
        // §6  Controller World Return Ordering (behavior, coverage #14)
        // ============================================================

        [Test]
        public void Controller_DoesNotConsumeQuestResultBeforeLastCompletion()
        {
            var go = new GameObject("ControllerOrderingTest");
            var worldGo = new GameObject("WorldEntryOrderingTest");
            try
            {
                var world = worldGo.AddComponent<PetRobotWorldEntryController>();
                var quest = go.AddComponent<PetRobotPortfolioQuestController>();
                var channel = new PetRobotWorldHandoffChannel();
                var transition = new FakeTransition();
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);

                var token = new PetRobotGreetingQuestToken();
                var prior = Mastered(9);
                channel.BeginGreetingQuest("W", "Practice", Vector3.zero, token, prior, DefaultIdentity());
                channel.CommitQuestReturn(new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.StrictSuccess, "ok", "", ""), prior);
                channel.TryConsumeCompleted("W", out _);

                quest.DriveUpdateLogic();
                Assert.IsTrue(channel.HasQuestResult,
                    "quest result should not be consumed while LastCompletion is null");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
                UnityEngine.Object.DestroyImmediate(worldGo);
            }
        }

        [Test]
        public void Controller_ConsumesQuestResultAfterLastCompletionAvailable()
        {
            var go = new GameObject("ControllerOrderingTest2");
            var worldGo = new GameObject("WorldEntryOrderingTest2");
            try
            {
                var world = worldGo.AddComponent<PetRobotWorldEntryController>();
                var quest = go.AddComponent<PetRobotPortfolioQuestController>();
                var channel = new PetRobotWorldHandoffChannel();
                var transition = new FakeTransition();
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);

                var token = new PetRobotGreetingQuestToken();
                var prior = Mastered(9);
                channel.BeginGreetingQuest("W", "Practice", Vector3.zero, token, prior, DefaultIdentity());
                channel.CommitQuestReturn(new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.StrictSuccess, "ok", "", ""), prior);
                channel.TryConsumeCompleted("W", out _);
                SetLastCompletion(world, prior);

                quest.DriveUpdateLogic();
                Assert.IsFalse(channel.HasQuestResult);
                Assert.AreEqual(PetRobotPortfolioQuestState.DemoCompleted, quest.State);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
                UnityEngine.Object.DestroyImmediate(worldGo);
            }
        }

        [Test]
        public void Controller_QuestResultConsumedExactlyOnce()
        {
            var go = new GameObject("ControllerIdempotentTest");
            var worldGo = new GameObject("WorldEntryIdempotentTest");
            try
            {
                var world = worldGo.AddComponent<PetRobotWorldEntryController>();
                var quest = go.AddComponent<PetRobotPortfolioQuestController>();
                var channel = new PetRobotWorldHandoffChannel();
                var transition = new FakeTransition();
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);

                var token = new PetRobotGreetingQuestToken();
                var prior = Mastered(9);
                channel.BeginGreetingQuest("W", "Practice", Vector3.zero, token, prior, DefaultIdentity());
                channel.CommitQuestReturn(new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.StrictSuccess, "ok", "", ""), prior);
                channel.TryConsumeCompleted("W", out _);
                SetLastCompletion(world, prior);

                quest.DriveUpdateLogic();
                Assert.AreEqual(PetRobotPortfolioQuestState.DemoCompleted, quest.State);
                quest.DriveUpdateLogic();
                Assert.AreEqual(PetRobotPortfolioQuestState.DemoCompleted, quest.State);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
                UnityEngine.Object.DestroyImmediate(worldGo);
            }
        }

        // ============================================================
        // §7  Transition Failure Rollback + Retry (behavior, coverage #5)
        // ============================================================

        [Test]
        public void Controller_TransitionFailure_RollsBackChannelAndFlow()
        {
            WithController((world, quest, channel) =>
            {
                var rejectTransition = new FakeTransition { AlwaysReject = true };
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, rejectTransition, "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsFalse(quest.CompleteDialogue(true));
                Assert.IsFalse(channel.HasPendingEntry);
                Assert.IsFalse(quest.QuestTransitionRequested);
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);
                Assert.IsNotEmpty(quest.LastQuestTransitionError);
            });
        }

        [Test]
        public void Controller_TransitionException_RollsBackChannelAndFlow()
        {
            WithController((world, quest, channel) =>
            {
                var throwTransition = new FakeTransition { ThrowOnRequest = true };
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, throwTransition, "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsFalse(quest.CompleteDialogue(true));
                Assert.IsFalse(channel.HasPendingEntry);
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);
                StringAssert.Contains("InvalidOperationException", quest.LastQuestTransitionError);
            });
        }

        [Test]
        public void Controller_TransitionSuccess_SetsQuestTransitionRequested()
        {
            WithController((world, quest, channel) =>
            {
                var fakeTransition = new FakeTransition();
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, fakeTransition, "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsTrue(quest.CompleteDialogue(true));
                Assert.IsTrue(quest.QuestTransitionRequested);
                Assert.IsTrue(channel.HasPendingEntry);
                Assert.AreEqual(string.Empty, quest.LastQuestTransitionError);
            });
        }

        [Test]
        public void Controller_TransitionFailure_RetrySucceedsAfterRollback()
        {
            WithController((world, quest, channel) =>
            {
                var fakeTransition = new FakeTransition { AlwaysReject = true };
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, fakeTransition, "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsFalse(quest.CompleteDialogue(true));
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);

                fakeTransition.AlwaysReject = false;
                Assert.IsTrue(quest.CompleteDialogue(true));
                Assert.IsTrue(quest.QuestTransitionRequested);
            });
        }

        // ============================================================
        // §8  Gate Precheck — Null = Fail-Closed + Retry (behavior, coverage #1 #5)
        // ============================================================

        [Test]
        public void Controller_GateNotReady_BlocksAndRollsBackToDialogue()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = false, Failure = "G=1 L=2 T=0" };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, new FakeTransition(), "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsFalse(quest.CompleteDialogue(true));
                Assert.IsFalse(quest.QuestTransitionRequested);
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);
                StringAssert.Contains("Gate not ready", quest.LastQuestTransitionError);
            });
        }

        [Test]
        public void Controller_GateReady_AllowsQuestTransition()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, new FakeTransition(), "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsTrue(quest.CompleteDialogue(true));
                Assert.IsTrue(quest.QuestTransitionRequested);
            });
        }

        [Test]
        public void Controller_NullGateVerifier_FailsClosed()
        {
            WithController((world, quest, channel) =>
            {
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, new FakeTransition(), "Practice", null);
                SetupQuestDialogueState(world, quest);

                Assert.IsFalse(quest.CompleteDialogue(true));
                Assert.IsFalse(quest.QuestTransitionRequested);
                StringAssert.Contains("fail-closed", quest.LastQuestTransitionError);
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);
            });
        }

        [Test]
        public void Controller_GateFailure_RetrySucceedsWhenReady()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = false, Failure = "not enrolled" };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, new FakeTransition(), "Practice", gate);
                SetupQuestDialogueState(world, quest);

                Assert.IsFalse(quest.CompleteDialogue(true));
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);

                gate.Ready = true;
                Assert.IsTrue(quest.CompleteDialogue(true));
                Assert.IsTrue(quest.QuestTransitionRequested);
            });
        }

        // ============================================================
        // §9  Controller production Awake creates verifier (behavior, coverage #1)
        // ============================================================

        [Test]
        public void Controller_Awake_CreatesProductionIdentityAndVerifier()
        {
            var go = new GameObject("AwakeVerifierTest");
            try
            {
                var quest = go.AddComponent<PetRobotPortfolioQuestController>();
                Assert.IsNotNull(quest.ProductionIdentity,
                    "Awake must create production identity");
                Assert.AreEqual("local_learner", quest.ProductionIdentity.LearnerId,
                    "default production identity must use local_learner");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        // ============================================================
        // §10  Typed Boundaries (behavior)
        // ============================================================

        [Test]
        public void IQuestHybridRecognitionSession_IsInterface()
        {
            Assert.IsTrue(typeof(IQuestHybridRecognitionSession).IsInterface);
        }

        [Test]
        public void IQuestGateVerifier_IsInterface()
        {
            Assert.IsTrue(typeof(IQuestGateVerifier).IsInterface);
        }

        [Test]
        public void SessionQuestConstructor_AcceptsIQuestHybridRecognitionSession()
        {
            var paramType = typeof(PetRobotSignPracticeSession)
                .GetConstructor(new[]
                {
                    typeof(IPersonalLivePracticeRecognitionAdapter),
                    typeof(IPersonalizationCandidateRepository),
                    typeof(IQuestHybridRecognitionSession),
                    typeof(PetRobotGreetingQuestToken),
                    typeof(IPetRobotCandidateCameraReadiness),
                });
            Assert.IsNotNull(paramType, "quest constructor should accept IQuestHybridRecognitionSession");
        }

        // ============================================================
        // §10b  Initialization TechnicalFailure Retry (behavior, coverage #12)
        // ============================================================

        [Test]
        public void Controller_InitTechFailure_SetsCanRetry()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = true };
                var transition = new FakeTransition();
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);
                SetupQuestDialogueState(world, quest);
                Assert.IsTrue(quest.CompleteDialogue(true));

                var token = channel.PendingQuestToken;
                var prior = Mastered(9);
                var tfResult = new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.TechnicalFailure, "init failed", "InitErr", "bootstrap crash");
                channel.CommitQuestReturn(tfResult, prior);
                channel.TryConsumeCompleted("W", out _);
                SetLastCompletion(world, prior);

                quest.DriveUpdateLogic();
                Assert.IsTrue(quest.CanRetryAfterInitializationFailure,
                    "after consuming TechnicalFailure result, retry should be available");
                Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingGreetingAttempt, quest.State);
            });
        }

        [Test]
        public void Controller_InitTechFailure_RetryGoesToDialogueOpen()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = true };
                var transition = new FakeTransition();
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);
                SetupQuestDialogueState(world, quest);
                Assert.IsTrue(quest.CompleteDialogue(true));

                var token = channel.PendingQuestToken;
                var prior = Mastered(9);
                var tfResult = new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.TechnicalFailure, "init failed", "InitErr", "bootstrap crash");
                channel.CommitQuestReturn(tfResult, prior);
                channel.TryConsumeCompleted("W", out _);
                SetLastCompletion(world, prior);
                quest.DriveUpdateLogic();

                Assert.IsTrue(quest.TryRetryAfterInitializationFailure());
                Assert.AreEqual(PetRobotPortfolioQuestState.DialogueOpen, quest.State);
                Assert.IsFalse(quest.CanRetryAfterInitializationFailure);
                StringAssert.Contains("initialization failure", quest.LastQuestTransitionError);
            });
        }

        [Test]
        public void Controller_InitTechFailure_RetryThenNewQuestSucceeds()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = true };
                var transition = new FakeTransition();
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);
                SetupQuestDialogueState(world, quest);
                Assert.IsTrue(quest.CompleteDialogue(true));

                var token = channel.PendingQuestToken;
                var prior = Mastered(9);
                var tfResult = new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.TechnicalFailure, "init failed", "InitErr", "bootstrap crash");
                channel.CommitQuestReturn(tfResult, prior);
                channel.TryConsumeCompleted("W", out _);
                SetLastCompletion(world, prior);
                quest.DriveUpdateLogic();
                Assert.IsTrue(quest.TryRetryAfterInitializationFailure());

                Assert.IsTrue(quest.CompleteDialogue(true));
                Assert.IsTrue(quest.QuestTransitionRequested);
                Assert.AreEqual(2, transition.RequestCount);
            });
        }

        [Test]
        public void Controller_SuccessResult_DoesNotSetCanRetry()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = true };
                var transition = new FakeTransition();
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, transition, "Practice", gate);
                SetupQuestDialogueState(world, quest);
                Assert.IsTrue(quest.CompleteDialogue(true));

                var token = channel.PendingQuestToken;
                var prior = Mastered(9);
                var okResult = new PetRobotGreetingQuestResult(token.Id,
                    TrustedHybridOutcomeKind.StrictSuccess, "ok", "", "");
                channel.CommitQuestReturn(okResult, prior);
                channel.TryConsumeCompleted("W", out _);
                SetLastCompletion(world, prior);
                quest.DriveUpdateLogic();

                Assert.IsFalse(quest.CanRetryAfterInitializationFailure,
                    "success result must not activate initialization retry");
            });
        }

        [Test]
        public void Controller_RetryWithoutTechFailure_Rejected()
        {
            WithController((world, quest, channel) =>
            {
                var gate = new FakeGateVerifier { Ready = true };
                quest.ConfigureForTests(world, null, null, null, 2.5f,
                    channel, new FakeTransition(), "Practice", gate);
                SetupQuestDialogueState(world, quest);
                Assert.IsFalse(quest.TryRetryAfterInitializationFailure());
            });
        }

        // ============================================================
        // §11  Presentation Fail-closed (string-based)
        // ============================================================

        [Test]
        public void GuidedPresentation_QuestMode_NullOutcome_DoesNotFallBackToFixedProfile()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/GuidedSignPracticePresentationController.cs");
            Assert.IsTrue(source.Contains("else if (!questModeActive)"),
                "quest mode should not fall through to fixed-profile outcome branch");
        }

        // ============================================================
        // §12  Bootstrap quest-only identity (string-based, coverage #2)
        // ============================================================

        [Test]
        public void BootstrapSource_WrapsSessionInAdapter()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs");
            Assert.IsTrue(source.Contains("TrustedHybridRecognitionSessionAdapter"));
        }

        [Test]
        public void BootstrapSource_QuestModeUsesOnlyHandoffIdentity()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs");
            int questModePos = source.IndexOf("IsQuestMode = true", StringComparison.Ordinal);
            int harnessPos = source.IndexOf("Resources.Load<GameObject>", StringComparison.Ordinal);
            Assert.IsTrue(questModePos > 0 && harnessPos > 0 && questModePos < harnessPos,
                "quest mode detection must happen before harness prefab load");
            Assert.IsTrue(source.Contains("questIdentity.HasExplicitRoot"),
                "quest mode must use handoff identity for repository root");
        }

        // ============================================================
        // §13  Watcher committed/transitioned (string-based, coverage #15)
        // ============================================================

        [Test]
        public void WatcherSource_UsesCommitQuestReturn()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs");
            Assert.IsTrue(source.Contains("CommitQuestReturn"));
        }

        [Test]
        public void WatcherSource_CommittedAndTransitionedAreSeparateStates()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs");
            Assert.IsTrue(source.Contains("committedQuestResult"),
                "watcher should track committed state separately from transitioned");
            Assert.IsTrue(source.Contains("RetryQuestTransition"),
                "watcher should have retry transition path");
        }

        // ============================================================
        // §14  UI shows transition error in DialogueOpen (string-based)
        // ============================================================

        [Test]
        public void UiPresenter_ShowsTransitionErrorInDialogueOpen()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestUiPresenter.cs");
            Assert.IsTrue(source.Contains("LastQuestTransitionError"),
                "DialogueOpen UI should display transition error");
        }

        // ============================================================
        // Test Doubles
        // ============================================================

        sealed class FakeTransition : IPetRobotWorldSceneTransition
        {
            public int RequestCount { get; private set; }
            public bool AlwaysReject { get; set; }
            public bool ThrowOnRequest { get; set; }

            public bool Request(string sceneName)
            {
                if (ThrowOnRequest)
                    throw new InvalidOperationException("test transition exception");
                if (AlwaysReject || string.IsNullOrWhiteSpace(sceneName))
                    return false;
                RequestCount++;
                return true;
            }
        }

        sealed class FakeGateVerifier : IQuestGateVerifier
        {
            public bool Ready { get; set; }
            public string Failure { get; set; } = string.Empty;
            public string LastFailure => Failure;

            public bool VerifyOverallEnrollmentGateReady() => Ready;
        }

        // ============================================================
        // Helpers
        // ============================================================

        static PetRobotPortfolioQuestFlow Observed(PetRobotPracticeCompletionSnapshot result)
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsTrue(flow.ObserveCompleted(result));
            return flow;
        }

        static PetRobotPracticeCompletionSnapshot Mastered(int candidates) =>
            new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                3, 0, candidates);

        static PetRobotLearnerRootIdentity DefaultIdentity() =>
            new PetRobotLearnerRootIdentity("local_learner");

        static void SetLastCompletion(
            PetRobotWorldEntryController controller,
            PetRobotPracticeCompletionSnapshot completion)
        {
            typeof(PetRobotWorldEntryController)
                .GetProperty("LastCompletion")
                .GetSetMethod(true)
                .Invoke(controller, new object[] { completion });
        }

        static void SetupQuestDialogueState(
            PetRobotWorldEntryController world,
            PetRobotPortfolioQuestController quest)
        {
            SetLastCompletion(world, Mastered(9));
            quest.CaptureCompletionFromWorldOnce();
            quest.Flow.SetNpcInRange(true);
            quest.TryOpenNpcDialogue(true);
        }

        delegate void ControllerTestBody(
            PetRobotWorldEntryController world,
            PetRobotPortfolioQuestController quest,
            PetRobotWorldHandoffChannel channel);

        static void WithController(ControllerTestBody body)
        {
            var go = new GameObject("QuestControllerTest");
            var worldGo = new GameObject("WorldEntryTest");
            try
            {
                var world = worldGo.AddComponent<PetRobotWorldEntryController>();
                var quest = go.AddComponent<PetRobotPortfolioQuestController>();
                var channel = new PetRobotWorldHandoffChannel();
                body(world, quest, channel);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
                UnityEngine.Object.DestroyImmediate(worldGo);
            }
        }

        static string ReadProject(string relativePath) => File.ReadAllText(
            Path.Combine(Directory.GetParent(Application.dataPath).FullName, relativePath))
            .Replace("\r\n", "\n");
    }
}
