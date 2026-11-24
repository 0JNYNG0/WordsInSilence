using System;
using System.IO;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotPortfolioQuestPresentationV1Tests
    {
        [Test]
        public void InitialState_ShowsPracticeObjective()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingPractice, flow.State);
            Assert.AreEqual("동행 애완 로봇과 대화해 수어 연습을 시작하세요 (E / Enter / Space)", flow.Objective);
            Assert.IsTrue(flow.CanEnterPractice);
        }

        [Test]
        public void CompletionNull_DoesNotAdvanceQuest()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsFalse(flow.ObserveCompleted(null));
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingPractice, flow.State);
        }

        [Test]
        public void IncompleteSession_DoesNotAdvanceToNpcReport()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsFalse(flow.ObserveIncompleteSession());
            Assert.IsNull(flow.Snapshot);
        }

        [Test]
        public void TechnicalFailure_DoesNotBecomeCompletedResult()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            Assert.IsFalse(flow.ObserveTechnicalFailure());
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingPractice, flow.State);
        }

        [Test]
        public void CompletedSnapshot_IsFrozenExactlyOnce()
        {
            var worldObject = new UnityEngine.GameObject("Step10CWorldEntryTest");
            var questObject = new UnityEngine.GameObject("Step10CQuestTest");
            try
            {
                var world = worldObject.AddComponent<PetRobotWorldEntryController>();
                var quest = questObject.AddComponent<PetRobotPortfolioQuestController>();
                quest.ConfigureForTests(world, null, null, null, 2.5f);

                SetLastCompletion(world, Mastered(8));
                Assert.IsTrue(quest.CaptureCompletionFromWorldOnce());
                Assert.IsFalse(world.enabled);

                SetLastCompletion(world, NeedsPractice(99));
                Assert.IsFalse(quest.CaptureCompletionFromWorldOnce());
                Assert.AreEqual(1, quest.Flow.CompletionFreezeCount);
                Assert.AreEqual(8, quest.Completion.CandidateSavedTotal);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(questObject);
                UnityEngine.Object.DestroyImmediate(worldObject);
            }
        }

        [Test]
        public void PortfolioQuest_DoesNotConsumeStep10BHandoffChannelForCompletion()
        {
            string source = ReadProject("Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestController.cs");
            Assert.IsFalse(source.Contains("TryConsumeCompleted"));
            int stateGuard = source.IndexOf(
                "flow.State != PetRobotPortfolioQuestState.AwaitingPractice", StringComparison.Ordinal);
            int snapshotGuard = source.IndexOf("flow.Snapshot != null", StringComparison.Ordinal);
            int completionRead = source.IndexOf(
                "PetRobotPracticeCompletionSnapshot completion = entryController.LastCompletion;",
                StringComparison.Ordinal);
            Assert.IsTrue(stateGuard >= 0 && snapshotGuard > stateGuard);
            Assert.IsTrue(completionRead > snapshotGuard);
        }

        [Test]
        public void CompletedSnapshot_PreservesThreeWordStatusesAndCounts()
        {
            var flow = Observed(NeedsPractice(8));
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, flow.Snapshot.Greeting);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, flow.Snapshot.Love);
            Assert.AreEqual(PetRobotWordLearningStatus.NeedsPractice, flow.Snapshot.ThankYou);
            Assert.AreEqual(2, flow.Snapshot.MasteredCount);
            Assert.AreEqual(1, flow.Snapshot.NeedsPracticeCount);
        }

        [Test]
        public void CompletedSnapshot_PreservesCandidateSavedTotal()
        {
            Assert.AreEqual(8, Observed(Mastered(8)).Snapshot.CandidateSavedTotal);
        }

        [Test]
        public void MasteredThree_UsesAllMasteredDialogueBranch()
        {
            var snapshot = Observed(Mastered(8)).Snapshot;
            StringAssert.Contains("세 수어를 모두 익혔습니다", snapshot.ResultHeadline);
            StringAssert.Contains("다음 임무", snapshot.ResultFollowUp);
        }

        [Test]
        public void NeedsPractice_UsesHonestReviewDialogueBranch()
        {
            var snapshot = Observed(NeedsPractice(8)).Snapshot;
            StringAssert.Contains("1개 수어는 복습이 필요합니다", snapshot.ResultHeadline);
            Assert.AreEqual(PetRobotWordLearningStatus.NeedsPractice,
                snapshot.GetWordStatus("THANK_YOU"));
        }

        [Test]
        public void FrozenSnapshot_BlocksAccidentalPracticeReentry()
        {
            var flow = Observed(Mastered(8));
            Assert.IsFalse(flow.CanEnterPractice);
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingNpcReport, flow.State);
        }

        [Test]
        public void NpcInteractionOutsideRange_IsBlocked()
        {
            var flow = Observed(Mastered(8));
            flow.SetNpcInRange(false);
            Assert.IsFalse(flow.TryOpenDialogue(true));
        }

        [Test]
        public void NpcInteractionInsideRange_ExposesPromptGate()
        {
            var flow = Observed(Mastered(8));
            flow.SetNpcInRange(true);
            Assert.IsTrue(flow.CanReportToNpc);
        }

        [Test]
        public void OneExplicitInput_OpensDialogueExactlyOnce()
        {
            var flow = Observed(Mastered(8));
            flow.SetNpcInRange(true);
            Assert.IsFalse(flow.TryOpenDialogue(false));
            Assert.IsTrue(flow.TryOpenDialogue(true));
            Assert.AreEqual(1, flow.DialogueOpenCount);
        }

        [Test]
        public void RepeatedInput_CannotDuplicateDialogueOrCompletionTransition()
        {
            var flow = Observed(Mastered(8));
            flow.SetNpcInRange(true);
            Assert.IsTrue(flow.TryOpenDialogue(true));
            Assert.IsFalse(flow.TryOpenDialogue(true));
            Assert.IsTrue(flow.TryCompleteDialogue(true));
            Assert.IsFalse(flow.TryCompleteDialogue(true));
            Assert.AreEqual(1, flow.DialogueOpenCount);
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingGreetingAttempt, flow.State);
            Assert.AreEqual(0, flow.DemoCompletionCount);
            Assert.IsTrue(flow.TryCompleteGreetingQuest());
            Assert.IsFalse(flow.TryCompleteGreetingQuest());
            Assert.AreEqual(1, flow.DemoCompletionCount);
        }

        [Test]
        public void NpcReportWithoutCompletion_IsBlocked()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            flow.SetNpcInRange(true);
            Assert.IsFalse(flow.TryOpenDialogue(true));
        }

        [Test]
        public void DemoCompleted_OnlyAfterNpcReportDialogueAndGreetingQuest()
        {
            var flow = Observed(Mastered(8));
            Assert.IsFalse(flow.TryCompleteDialogue(true));
            flow.SetNpcInRange(true);
            Assert.IsTrue(flow.TryOpenDialogue(true));
            Assert.IsTrue(flow.TryCompleteDialogue(true));
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingGreetingAttempt, flow.State);
            Assert.IsTrue(flow.TryCompleteGreetingQuest());
            Assert.AreEqual(PetRobotPortfolioQuestState.DemoCompleted, flow.State);
        }

        [Test]
        public void PortfolioWorldScene_HasNoWebcamProviderOrCandidateCameraOwner()
        {
            string scene = ReadProject("Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity");
            Assert.IsFalse(scene.Contains("WebCam"));
            Assert.IsFalse(scene.Contains("HandLandmarker"));
            Assert.IsFalse(scene.Contains("PoseLandmarker"));
            Assert.IsFalse(scene.Contains("candidateCameraRoute"));
            Assert.IsFalse(scene.Contains("e33729c5ad6f4e44ad02aa19b2710d40"));
        }

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

        static PetRobotPracticeCompletionSnapshot NeedsPractice(int candidates) =>
            new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.NeedsPractice,
                2, 1, candidates);

        static void SetLastCompletion(
            PetRobotWorldEntryController controller,
            PetRobotPracticeCompletionSnapshot completion)
        {
            typeof(PetRobotWorldEntryController)
                .GetProperty("LastCompletion")
                .GetSetMethod(true)
                .Invoke(controller, new object[] { completion });
        }

        static int CountOccurrences(string source, string token)
        {
            int count = 0;
            int offset = 0;
            while ((offset = source.IndexOf(token, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += token.Length;
            }
            return count;
        }

        static string ReadProject(string relativePath) => File.ReadAllText(
            Path.Combine(Directory.GetParent(UnityEngine.Application.dataPath).FullName, relativePath))
            .Replace("\r\n", "\n");
    }
}
