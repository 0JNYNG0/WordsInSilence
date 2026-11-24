using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotWorldEntryReturnIntegrationV1Tests
    {
        const string World = "PetRobotWorldEntryReturnVerticalSlice";
        const string Practice = "PetRobotSignPracticeVerticalSlice";

        [Test]
        public void InteractionOutsideRange_IsBlocked()
        {
            var transition = new FakeTransition();
            var gate = Gate(new PetRobotWorldHandoffChannel(), transition, null);
            gate.SetPlayerInRange(false);
            Assert.IsFalse(gate.TryEnter(true, new Vector3(1f, 0f, 2f)));
            Assert.AreEqual(0, transition.RequestCount);
        }

        [Test]
        public void InteractionInsideRange_ExposesManualEntryGate()
        {
            var gate = Gate(new PetRobotWorldHandoffChannel(), new FakeTransition(), null);
            gate.SetPlayerInRange(true);
            Assert.IsTrue(gate.PlayerInRange);
            Assert.IsFalse(gate.TransitionRequested);
        }

        [Test]
        public void EntryRequiresOneExplicitInteractionPress()
        {
            var transition = new FakeTransition();
            var gate = Gate(new PetRobotWorldHandoffChannel(), transition, null);
            gate.SetPlayerInRange(true);
            Assert.IsFalse(gate.TryEnter(false, Vector3.zero));
            Assert.IsTrue(gate.TryEnter(true, Vector3.zero));
            Assert.AreEqual(1, transition.RequestCount);
        }

        [Test]
        public void RepeatedInteraction_CannotDuplicateSceneTransition()
        {
            var transition = new FakeTransition();
            var gate = Gate(new PetRobotWorldHandoffChannel(), transition, null);
            gate.SetPlayerInRange(true);
            Assert.IsTrue(gate.TryEnter(true, Vector3.zero));
            Assert.IsFalse(gate.TryEnter(true, Vector3.zero));
            Assert.IsFalse(gate.TryEnter(true, Vector3.zero));
            Assert.AreEqual(1, transition.RequestCount);
        }

        [Test]
        public void PracticeEntry_DoesNotAutomaticallyStartCaptureOrRetry()
        {
            string watcher = ReadProject("Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs");
            string controller = ReadProject("Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs");
            Assert.IsFalse(watcher.Contains("StartExpression("));
            Assert.IsFalse(watcher.Contains("BeginCapture("));
            Assert.IsFalse(watcher.Contains("Retry("));
            Assert.IsFalse(controller.Contains("StartExpression("));
            Assert.IsFalse(controller.Contains("BeginCapture("));
        }

        [Test]
        public void ApprovedPracticeScene_KeepsExactlyOneCandidateCameraOwner()
        {
            string scene = ReadProject("Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity");
            Assert.AreEqual(1, Count(scene, "guid: e33729c5ad6f4e44ad02aa19b2710d40"));
            Assert.AreEqual(1, Count(scene, "candidateCameraRoute: {fileID: 1669740324}"));
        }

        [Test]
        public void IncompleteSession_CannotPublishCompletedReturnPayload()
        {
            var channel = new PetRobotWorldHandoffChannel();
            Assert.IsTrue(channel.Begin(World, Practice, new Vector3(3f, 0f, 4f)));
            PetRobotWorldReturnPayload payload;
            Assert.IsFalse(channel.TryConsumeCompleted(World, out payload));
            Assert.IsFalse(channel.HasPublishedCompletion);
        }

        [Test]
        public void TechnicalFailureWithoutCompletedSnapshot_CannotReturnAsCompletion()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var transition = new FakeTransition();
            Assert.IsTrue(channel.Begin(World, Practice, Vector3.zero));
            var gate = new PetRobotPracticeReturnGate(World, channel, transition);
            Assert.IsFalse(gate.TryReturn(true));
            Assert.AreEqual(0, transition.RequestCount);
            Assert.IsFalse(channel.HasPublishedCompletion);
        }

        [Test]
        public void CompletedSnapshot_PreservesThreeMasteredStatusesAndCounts()
        {
            PetRobotPracticeCompletionSnapshot value = Mastered(5);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, value.Greeting);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, value.Love);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, value.ThankYou);
            Assert.AreEqual(3, value.MasteredCount);
            Assert.AreEqual(0, value.NeedsPracticeCount);
            Assert.AreEqual(5, value.CandidateSavedTotal);
        }

        [Test]
        public void CompletedSnapshot_PreservesNeedsPracticeWithoutReinterpretingSuccess()
        {
            var value = new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.NeedsPractice,
                2, 1, 5);
            Assert.AreEqual(PetRobotWordLearningStatus.NeedsPractice, value.ThankYou);
            Assert.AreEqual(2, value.MasteredCount);
            Assert.AreEqual(1, value.NeedsPracticeCount);
            StringAssert.Contains("1개 복습 필요", value.WorldSummaryMessage);
        }

        [Test]
        public void ExitButtonContract_RemainsCompletedOnly()
        {
            string session = ReadProject("Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs");
            string ui = ReadProject("Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeUiPresenter.cs");
            StringAssert.Contains("if (State != PetRobotPracticeState.Completed) return false;", session);
            StringAssert.Contains("session.State == PetRobotPracticeState.Completed && GUILayout.Button(\"연습 공간 나가기\")", ui);
        }

        [Test]
        public void CompletedExit_ReturnsToWorldExactlyOnce()
        {
            var channel = new PetRobotWorldHandoffChannel();
            var transition = new FakeTransition();
            Assert.IsTrue(channel.Begin(World, Practice, new Vector3(2f, 0f, 7f)));
            var gate = new PetRobotPracticeReturnGate(World, channel, transition);
            Assert.IsTrue(gate.ObserveCompleted(Mastered(5)));
            Assert.IsFalse(gate.TryReturn(false));
            Assert.IsTrue(gate.TryReturn(true));
            Assert.IsFalse(gate.TryReturn(true));
            Assert.AreEqual(1, transition.RequestCount);
        }

        [Test]
        public void CompletedResult_IsConsumedExactlyOnceInWorld()
        {
            var channel = PublishedChannel(Mastered(5), new Vector3(4f, 0f, -2f));
            PetRobotWorldReturnPayload first;
            PetRobotWorldReturnPayload second;
            Assert.IsTrue(channel.TryConsumeCompleted(World, out first));
            Assert.IsFalse(channel.TryConsumeCompleted(World, out second));
            Assert.AreEqual(new Vector3(4f, 0f, -2f), first.PlayerPosition);
        }

        [Test]
        public void WorldSummary_DistinguishesMasteredFromNeedsPractice()
        {
            Assert.AreEqual("연습 완료", Mastered(5).WorldSummaryMessage);
            var needs = new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.NeedsPractice,
                PetRobotWordLearningStatus.Mastered,
                2, 1, 5);
            Assert.AreEqual("연습 완료 · 1개 복습 필요", needs.WorldSummaryMessage);
        }

        [Test]
        public void Reentry_UsesFreshTransientGateAfterOneShotResultConsumption()
        {
            var channel = PublishedChannel(Mastered(5), Vector3.zero);
            PetRobotWorldReturnPayload payload;
            Assert.IsTrue(channel.TryConsumeCompleted(World, out payload));
            var transition = new FakeTransition();
            var nextGate = Gate(channel, transition, null);
            nextGate.SetPlayerInRange(true);
            Assert.IsTrue(nextGate.TryEnter(true, Vector3.zero));
            Assert.AreEqual(1, transition.RequestCount);
        }

        [Test]
        public void WorldSceneAndPrefab_UsePlaceholderPlayerAndReferencedVendorEnvironmentOnly()
        {
            string scene = ReadProject("Assets/_Project/Scenes/VerticalSlice/PetRobotWorldEntryReturnVerticalSlice.unity");
            string prefab = ReadProject("Assets/_Project/Resources/VerticalSlice/PetRobotWorldEntryReturn.prefab");
            StringAssert.Contains("guid: 10b00000000000000000000000000010", scene);
            StringAssert.Contains("guid: 5c0d962245381c74f8e0e76dd95e4c2c", scene);
            StringAssert.Contains("--- !u!143 &14300001", prefab);
            StringAssert.Contains("guid: 10b00000000000000000000000000003", prefab);
            StringAssert.Contains("guid: 10b00000000000000000000000000004", prefab);
            Assert.IsFalse(scene.Contains("PracticeController"));
            Assert.IsFalse(prefab.Contains("WebCam"));
        }

        [Test]
        public void BuildSettings_ContainsWorldAndApprovedPracticeScenesOnceEach()
        {
            string settings = ReadProject("ProjectSettings/EditorBuildSettings.asset");
            Assert.AreEqual(1, Count(settings,
                "Assets/_Project/Scenes/VerticalSlice/PetRobotWorldEntryReturnVerticalSlice.unity"));
            Assert.AreEqual(1, Count(settings,
                "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"));
        }

        [Test]
        public void WorldIntegration_DoesNotMutateRecognitionCandidateOrProfileAssets()
        {
            string directory = ProjectPath("Assets/_Project/Scripts/Practice/PetRobotWorldIntegration");
            string combined = string.Join("\n", Directory.GetFiles(directory, "*.cs")
                .OrderBy(value => value, StringComparer.Ordinal)
                .Select(File.ReadAllText));
            Assert.IsFalse(combined.Contains("MaskAwareDtw"));
            // the approved gate may READ the trusted repository (ProductionQuestGateVerifier
            // .ReloadTrustedAndVerify predates this work), so a source-text ban on the type name
            // is invalid; asset preservation is verified with pre/post storage inventories
            Assert.IsFalse(combined.Contains("personal_multi_prototype_threshold_config"));
            Assert.IsFalse(combined.Contains(".wisrt"));
            Assert.IsFalse(combined.Contains("SetExpectedWord("));
        }

        static PetRobotWorldEntryGate Gate(
            PetRobotWorldHandoffChannel channel,
            FakeTransition transition,
            Action prepare)
        {
            return new PetRobotWorldEntryGate(
                World, Practice, channel, transition, prepare ?? (() => { }));
        }

        static PetRobotPracticeCompletionSnapshot Mastered(int candidates) =>
            new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                3, 0, candidates);

        static PetRobotWorldHandoffChannel PublishedChannel(
            PetRobotPracticeCompletionSnapshot completion,
            Vector3 position)
        {
            var channel = new PetRobotWorldHandoffChannel();
            Assert.IsTrue(channel.Begin(World, Practice, position));
            Assert.IsTrue(channel.PublishCompleted(completion));
            return channel;
        }

        static string ProjectPath(string relativePath) =>
            Path.Combine(Directory.GetParent(Application.dataPath).FullName, relativePath);

        static string ReadProject(string relativePath) =>
            File.ReadAllText(ProjectPath(relativePath)).Replace("\r\n", "\n");

        static int Count(string text, string value)
        {
            int count = 0;
            int offset = 0;
            while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += value.Length;
            }
            return count;
        }

        sealed class FakeTransition : IPetRobotWorldSceneTransition
        {
            public int RequestCount { get; private set; }
            public bool Request(string sceneName)
            {
                RequestCount++;
                return string.Equals(sceneName, Practice, StringComparison.Ordinal) ||
                    string.Equals(sceneName, World, StringComparison.Ordinal);
            }
        }
    }
}
