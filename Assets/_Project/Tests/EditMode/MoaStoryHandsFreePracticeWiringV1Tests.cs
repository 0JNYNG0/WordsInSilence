using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Calibration;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class MoaStoryHandsFreePracticeWiringV1Tests
    {
        const string PresentationSource =
            "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/GuidedSignPracticePresentationController.cs";
        const string StorySource =
            "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/MoaStoryGuidedPracticeFlow.cs";
        const string DemonstrationSource =
            "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/SignDemonstrationAvatarPresenter.cs";
        const string BootstrapSource =
            "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs";

        [Test]
        public void Welcome_NoDemonstrationAndNoBeginAutoAttempt()
        {
            MoaStoryGuidedPracticeFlow flow = NewReadyFlow();
            Assert.AreEqual(MoaStoryBeat.Welcome, flow.Beat);
            Assert.AreEqual(MoaStoryAction.AdvanceStory, flow.Current.Action);
            Assert.AreNotEqual(MoaStoryAction.BeginAutoAttempt, flow.Current.Action);
        }

        [Test]
        public void LessonOverview_NoDemonstrationAndNoBeginAutoAttempt()
        {
            MoaStoryGuidedPracticeFlow flow = NewReadyFlow();
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.LessonOverview, flow.Beat);
            Assert.AreNotEqual(MoaStoryAction.BeginAutoAttempt, flow.Current.Action);
        }

        [Test]
        public void HowToUse_UsesHandsFreeCopyAndAdvancesToWordIntroduction()
        {
            MoaStoryGuidedPracticeFlow flow = NewReadyFlow();
            flow.AdvanceStory();
            flow.AdvanceStory();
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.HowToUseCopy, flow.Current.BodyText);
            StringAssert.Contains("자동으로 시작", flow.Current.BodyText);
            StringAssert.Contains("자동으로 끝낼게", flow.Current.BodyText);
            Assert.AreEqual("첫 표현 보기", flow.Current.ActionLabel);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.WordIntroduction, flow.Beat);
        }

        [Test]
        public void WordIntroduction_ActionIsBeginAutoAttempt()
        {
            MoaStoryGuidedPracticeFlow flow = ReachWordIntroduction(NewReadyFlow());
            Assert.AreEqual("직접 해볼게", flow.Current.ActionLabel);
            Assert.AreEqual(MoaStoryAction.BeginAutoAttempt, flow.Current.Action);
            Assert.IsTrue(flow.Current.ActionEnabled);
        }

        [Test]
        public void BeginAutoAttempt_RejectedRequestDoesNotAdvanceOrPause()
        {
            using (var rig = new WiringRig(PersonalLiveCaptureState.Capturing))
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                Assert.AreEqual(MoaStoryBeat.WordIntroduction, rig.Flow.Beat);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Inactive, rig.Coordinator.State);
                Assert.Zero(rig.Adapter.BeginCalls);
                Assert.Zero(rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void BeginAutoAttempt_AcceptedRequestsCoordinatorExactlyOnceAndEntersPractice()
        {
            using (var rig = new WiringRig())
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                Assert.AreEqual(MoaStoryBeat.Practice, rig.Flow.Beat);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration,
                    rig.Coordinator.State);
                Assert.Zero(rig.Adapter.BeginCalls);
                Assert.Zero(rig.Repository.SaveCalls);

                string source = Read(PresentationSource);
                Assert.AreEqual(1, Count(MethodBody(source, "void HandleBeginAutoAttempt()"),
                    "RequestCoordinatorAttempt()"));
                Assert.AreEqual(1, Count(MethodBody(source, "bool RequestCoordinatorAttempt()"),
                    "neutralAutoCaptureCoordinator.RequestAttempt()"));
            }
        }

        [Test]
        public void NormalBeginPath_DoesNotCallSessionStartExpression()
        {
            string body = MethodBody(Read(PresentationSource), "void HandleBeginAutoAttempt()");
            Assert.Zero(Count(body, "StartExpression"));
            StringAssert.Contains("RequestCoordinatorAttempt()", body);
        }

        [TestCase(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration,
            MoaStoryGuidedPracticeFlow.CalibrationCopy, MoaStoryAction.None)]
        [TestCase(RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral,
            MoaStoryGuidedPracticeFlow.CalibrationCopy, MoaStoryAction.None)]
        [TestCase(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral,
            MoaStoryGuidedPracticeFlow.ReadyCopy, MoaStoryAction.None)]
        [TestCase(RawMediaPipeNeutralAutoSegmenterState.Armed,
            MoaStoryGuidedPracticeFlow.ArmedCopy, MoaStoryAction.None)]
        [TestCase(RawMediaPipeNeutralAutoSegmenterState.Capturing,
            MoaStoryGuidedPracticeFlow.CapturingCopy, MoaStoryAction.None)]
        [TestCase(RawMediaPipeNeutralAutoSegmenterState.EndCandidate,
            MoaStoryGuidedPracticeFlow.EndCandidateCopy, MoaStoryAction.None)]
        [TestCase(RawMediaPipeNeutralAutoSegmenterState.Evaluating,
            MoaStoryGuidedPracticeFlow.EvaluatingCopy, MoaStoryAction.None)]
        public void AutoPhaseGuidance_HasExpectedCopyAndNoManualAction(
            RawMediaPipeNeutralAutoSegmenterState state,
            string expectedCopy,
            MoaStoryAction expectedAction)
        {
            PetRobotPracticeState sessionState =
                state == RawMediaPipeNeutralAutoSegmenterState.Capturing ||
                state == RawMediaPipeNeutralAutoSegmenterState.EndCandidate ||
                state == RawMediaPipeNeutralAutoSegmenterState.Evaluating
                    ? PetRobotPracticeState.Capturing
                    : PetRobotPracticeState.Ready;
            MoaStoryGuidedPracticeFlow flow = ReachPractice(NewReadyFlow());
            flow.Synchronize("GREETING", sessionState, MoaStoryFeedbackKind.None,
                false, true, state, true);
            Assert.AreEqual(expectedCopy, flow.Current.BodyText);
            Assert.AreEqual(expectedAction, flow.Current.Action);
            Assert.IsFalse(flow.Current.ActionVisible);
        }

        [Test]
        public void CalibrationNeutralAndEndHold_UseOnlySmallProgressStatus()
        {
            string source = Read(PresentationSource);
            string status = MethodBody(source, "string AutoPhaseStatusText()");
            StringAssert.Contains("CalibrationProgress", status);
            StringAssert.Contains("NeutralHoldProgress", status);
            StringAssert.Contains("EndHoldProgress", status);
            Assert.Zero(Count(status, "RequestAttempt"));
            Assert.Zero(Count(status, "FinishExpression"));
        }

        [Test]
        public void PreCaptureFailure_ShowsRetryActionWithoutLearnerOutcomeOrCandidate()
        {
            using (var rig = new WiringRig())
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                rig.Source.EmitFailure(new InvalidOperationException("generic pre-capture failure"));
                Assert.AreEqual(PetRobotPracticeState.Ready, rig.Session.State);
                Assert.IsNull(rig.Session.LastLearningOutcome);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, rig.Coordinator.State);
                Assert.AreEqual(MoaStoryGuidedPracticeFlow.PreCaptureFailureCopy,
                    rig.Flow.Current.BodyText);
                Assert.AreEqual(MoaStoryAction.RetryPreCapture, rig.Flow.Current.Action);
                Assert.AreEqual("다시 준비할게", rig.Flow.Current.ActionLabel);
                Assert.Zero(rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void PreCaptureRetry_CallsRequestAttemptOnlyAndReentersNeutralFlow()
        {
            using (var rig = new WiringRig())
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                rig.Source.EmitFailure(new InvalidOperationException("generic pre-capture failure"));
                InvokePrivate(rig.Controller, "HandlePreCaptureRetry");
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration,
                    rig.Coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Ready, rig.Session.State);
                Assert.IsNull(rig.Session.LastLearningOutcome);
                Assert.Zero(rig.Adapter.ResetCalls);
                Assert.Zero(rig.Repository.SaveCalls);

                string body = MethodBody(Read(PresentationSource), "void HandlePreCaptureRetry()");
                StringAssert.Contains("RequestCoordinatorAttempt()", body);
                Assert.Zero(Count(body, "session.Retry"));
                Assert.Zero(Count(body, "FinishExpression"));
            }
        }

        [Test]
        public void ActiveTechnicalFailure_RetriesSessionThenRequestsOneFreshAttempt()
        {
            using (var rig = new WiringRig())
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                rig.Adapter.NextAttempt = Attempt(GuidedQuestOutcomeKind.TechnicalFailure, "GREETING");
                Assert.IsTrue(rig.Session.StartExpression());
                Assert.IsTrue(rig.Session.FinishExpression());
                Assert.AreEqual(MoaStoryAction.RetryTechnicalFailure, rig.Flow.Current.Action);
                InvokePrivate(rig.Controller, "HandleTechnicalFailureRetry");
                Assert.AreEqual(1, rig.Adapter.ResetCalls);
                Assert.AreEqual(PetRobotPracticeState.Ready, rig.Session.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration,
                    rig.Coordinator.State);
                Assert.Zero(rig.Repository.SaveCalls);

                string body = MethodBody(Read(PresentationSource), "void HandleTechnicalFailureRetry()");
                Assert.Less(body.IndexOf("session.Retry()", StringComparison.Ordinal),
                    body.IndexOf("RequestCoordinatorAttempt()", StringComparison.Ordinal));
                Assert.AreEqual(1, Count(body, "RequestCoordinatorAttempt()"));
            }
        }

        [Test]
        public void StrictSuccessAndLowConfidence_PreserveExistingNextSemantics()
        {
            foreach (MoaStoryFeedbackKind kind in new[]
                     {
                         MoaStoryFeedbackKind.StrictSuccess,
                         MoaStoryFeedbackKind.LowConfidenceCorrect
                     })
            {
                MoaStoryGuidedPracticeFlow flow = ReachPractice(NewReadyFlow());
                flow.Synchronize("GREETING", PetRobotPracticeState.Feedback, kind,
                    false, true, RawMediaPipeNeutralAutoSegmenterState.Evaluating, true);
                Assert.AreEqual(MoaStoryAction.ContinueToNextWord, flow.Current.Action);
            }
        }

        [Test]
        public void FirstWrongWord_RequiresReviewBeforeAnotherExplicitAttempt()
        {
            using (var rig = new WiringRig())
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                rig.Adapter.NextAttempt = Attempt(GuidedQuestOutcomeKind.WrongWord, "GREETING");
                Assert.IsTrue(rig.Session.StartExpression());
                Assert.IsTrue(rig.Session.FinishExpression());
                Assert.AreEqual(MoaStoryAction.ReviewWrongWord, rig.Flow.Current.Action);
                InvokePrivate(rig.Controller, "HandleWrongWordReview");
                Assert.AreEqual(MoaStoryBeat.WordIntroduction, rig.Flow.Beat);
                Assert.AreEqual(MoaStoryAction.BeginAutoAttempt, rig.Flow.Current.Action);
                Assert.AreEqual(PetRobotPracticeState.Ready, rig.Session.State);
                Assert.Zero(rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void NeedsPracticeAndCompleted_KeepExistingContinuationAndExitActions()
        {
            MoaStoryGuidedPracticeFlow flow = ReachPractice(NewReadyFlow());
            flow.Synchronize("GREETING", PetRobotPracticeState.Feedback,
                MoaStoryFeedbackKind.WrongWord, true, true,
                RawMediaPipeNeutralAutoSegmenterState.Evaluating, true);
            Assert.AreEqual(MoaStoryAction.ContinueWithNeedsPractice, flow.Current.Action);
            flow.Synchronize("THANK_YOU", PetRobotPracticeState.Completed,
                MoaStoryFeedbackKind.StrictSuccess, false, true,
                RawMediaPipeNeutralAutoSegmenterState.Evaluating, true);
            Assert.AreEqual(MoaStoryAction.ExitPractice, flow.Current.Action);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.CompletedCopy, flow.Current.BodyText);
        }

        [Test]
        public void NormalStoryUi_HidesAllLegacyControlsAndTechnicalCandidateMetrics()
        {
            using (var rig = new WiringRig())
            {
                foreach (string fieldName in new[]
                         {
                             "startButton", "finishButton", "retryButton", "needsPracticeButton",
                             "nextButton", "exitButton"
                         })
                    Assert.IsFalse(GetField<Button>(rig.Controller, fieldName).gameObject.activeSelf,
                        fieldName);
                Assert.IsFalse(GetField<Text>(rig.Controller, "candidateText").gameObject.activeSelf);
                Assert.IsFalse(GetField<Text>(rig.Controller, "resultText").gameObject.activeSelf);
                Assert.IsTrue(GetField<Button>(rig.Controller, "storyActionButton").gameObject.activeSelf);
            }
        }

        [Test]
        public void Bootstrap_PassesTheAlreadyCreatedCoordinatorToPresentation()
        {
            string bootstrap = Read(BootstrapSource);
            string presentation = Read(PresentationSource);
            string bind = MethodBody(bootstrap, "void Start()");
            StringAssert.Contains("neutralAutoCaptureCoordinator = new PetRobotNeutralAutoCaptureCoordinator(", bind);
            StringAssert.Contains("guidedPresentation.Bind(", bind);
            StringAssert.Contains("neutralAutoCaptureCoordinator);", bind);
            Assert.Zero(Count(presentation, "new PetRobotNeutralAutoCaptureCoordinator"));
            Assert.Zero(Count(presentation, "new RawMediaPipeNeutralAutoSegmenter"));
            Assert.Zero(Count(presentation, "new PetRobotNeutralCalibrationProfileService"));
            Assert.Zero(Count(presentation, "new PersonalLivePracticeRecognitionAdapter"));
        }

        [Test]
        public void BindRebindAndDestroy_SubscribeAndUnsubscribeSessionAndCoordinatorExactlyOnce()
        {
            using (var rig = new WiringRig())
            {
                Assert.AreEqual(1, EventSubscriberCount(rig.Session, "Changed"));
                Assert.AreEqual(1, EventSubscriberCount(rig.Coordinator, "Changed"));
                rig.Controller.Bind(rig.Session, null, null, null, rig.Coordinator);
                Assert.AreEqual(1, EventSubscriberCount(rig.Session, "Changed"));
                Assert.AreEqual(1, EventSubscriberCount(rig.Coordinator, "Changed"));
                InvokePrivate(rig.Controller, "OnDestroy");
                rig.DestroyController();
                Assert.Zero(EventSubscriberCount(rig.Session, "Changed"));
                Assert.Zero(EventSubscriberCount(rig.Coordinator, "Changed"));
            }
        }

        [Test]
        public void Demonstration_IsHiddenBeforeWordIntroductionAndStartsThere()
        {
            using (var stage = new StageRig())
            using (var rig = new WiringRig(PersonalLiveCaptureState.Ready, stage.Presenter))
            {
                Assert.AreEqual(string.Empty, stage.Presenter.CurrentWord);
                rig.Flow.AdvanceStory();
                InvokePrivate(rig.Controller, "RefreshUgui");
                Assert.AreEqual(string.Empty, stage.Presenter.CurrentWord);
                rig.Flow.AdvanceStory();
                InvokePrivate(rig.Controller, "RefreshUgui");
                Assert.AreEqual(string.Empty, stage.Presenter.CurrentWord);
                rig.Flow.AdvanceStory();
                InvokePrivate(rig.Controller, "RefreshUgui");
                Assert.AreEqual("GREETING", stage.Presenter.CurrentWord);
                Assert.IsFalse(stage.Presenter.DemonstrationPlaybackPaused);
                Assert.AreEqual(0d, stage.Presenter.CurrentPlayableTime, 1e-6d);
            }
        }

        [Test]
        public void AcceptedAttempt_PausesResetsAndPausedClockDoesNotAdvance()
        {
            using (var stage = new StageRig())
            using (var rig = new WiringRig(PersonalLiveCaptureState.Ready, stage.Presenter))
            {
                rig.ReachWordIntroduction();
                stage.Presenter.AdvanceDemonstrationPresentation(0.25f);
                Assert.Greater(stage.Presenter.CurrentPlayableTime, 0d);
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                Assert.IsTrue(stage.Presenter.DemonstrationPlaybackPaused);
                Assert.AreEqual(0d, stage.Presenter.CurrentPlayableTime, 1e-6d);
                stage.Presenter.AdvanceDemonstrationPresentation(0.5f);
                Assert.AreEqual(0d, stage.Presenter.CurrentPlayableTime, 1e-6d);
            }
        }

        [Test]
        public void WrongWordReview_ResumesSameWordDemonstration()
        {
            using (var stage = new StageRig())
            using (var rig = new WiringRig(PersonalLiveCaptureState.Ready, stage.Presenter))
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                rig.Adapter.NextAttempt = Attempt(GuidedQuestOutcomeKind.WrongWord, "GREETING");
                rig.Session.StartExpression();
                rig.Session.FinishExpression();
                InvokePrivate(rig.Controller, "HandleWrongWordReview");
                Assert.AreEqual("GREETING", stage.Presenter.CurrentWord);
                Assert.IsFalse(stage.Presenter.DemonstrationPlaybackPaused);
                Assert.AreEqual(0d, stage.Presenter.CurrentPlayableTime, 1e-6d);
            }
        }

        [Test]
        public void NextWordIntroduction_SwitchesAndLoopsTheNewDemonstration()
        {
            using (var stage = new StageRig())
            using (var rig = new WiringRig(PersonalLiveCaptureState.Ready, stage.Presenter))
            {
                rig.ReachWordIntroduction();
                InvokePrivate(rig.Controller, "HandleBeginAutoAttempt");
                rig.Adapter.NextAttempt = Attempt(GuidedQuestOutcomeKind.StrictSuccess, "GREETING");
                rig.Session.StartExpression();
                rig.Session.FinishExpression();
                InvokePrivate(rig.Controller, "HandleNextWord");
                Assert.AreEqual("LOVE", rig.Session.CurrentWord);
                Assert.AreEqual(MoaStoryBeat.WordIntroduction, rig.Flow.Beat);
                Assert.AreEqual("LOVE", stage.Presenter.CurrentWord);
                Assert.IsFalse(stage.Presenter.DemonstrationPlaybackPaused);
                Assert.AreEqual(0d, stage.Presenter.CurrentPlayableTime, 1e-6d);
            }
        }

        [Test]
        public void SerializedDemonstrationAndRobotTransformsRemainUnchanged()
        {
            using (var stage = new StageRig())
            {
                Assert.AreEqual(new Vector3(-6.65f, -3.2f, -0.65f),
                    stage.Presenter.DemonstrationWorldPosition);
                Assert.AreEqual(new Vector3(5.85f, 1.95f, 0.35f),
                    stage.Presenter.RobotWorldPosition);
                Assert.AreEqual(new Vector3(3.25f, 3.25f, 3.25f),
                    GetValue<Vector3>(stage.Presenter, "demonstrationScale"));
                Assert.AreEqual(180f, GetValue<float>(stage.Presenter, "demonstrationYaw"));
                Assert.AreEqual(80f, GetValue<float>(stage.Presenter, "demonstrationYawSpeed"));
                Assert.AreEqual(new Vector3(0.68f, 0.68f, 0.68f),
                    GetValue<Vector3>(stage.Presenter, "robotScale"));
                Assert.AreEqual(220f, GetValue<float>(stage.Presenter, "robotYaw"));
            }
        }

        [Test]
        public void StoryFlow_RemainsPresentationOnlyAndCoordinatorCallsStayInController()
        {
            string story = Read(StorySource);
            string presentation = Read(PresentationSource);
            foreach (string forbidden in new[]
                     {
                         "RecognitionCore", "MaskAwareDtw", "PersonalizationCandidateRepository",
                         "PetRobotNeutralCalibrationProfileService", "StartExpression(",
                         "FinishExpression(", "RequestAttempt("
                     })
                Assert.Zero(Count(story, forbidden), forbidden);
            StringAssert.Contains("neutralAutoCaptureCoordinator.RequestAttempt()", presentation);
            Assert.Zero(Count(Read(DemonstrationSource), "RecognitionCore"));
        }

        static MoaStoryGuidedPracticeFlow NewReadyFlow()
        {
            var flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true,
                RawMediaPipeNeutralAutoSegmenterState.Inactive, true);
            return flow;
        }

        static MoaStoryGuidedPracticeFlow ReachWordIntroduction(MoaStoryGuidedPracticeFlow flow)
        {
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            return flow;
        }

        static MoaStoryGuidedPracticeFlow ReachPractice(MoaStoryGuidedPracticeFlow flow)
        {
            ReachWordIntroduction(flow);
            Assert.IsTrue(flow.ConfirmAutoAttemptAccepted());
            return flow;
        }

        static PracticeRecognitionAttempt Attempt(GuidedQuestOutcomeKind kind, string expectedWord)
        {
            string predicted = kind == GuidedQuestOutcomeKind.WrongWord ? "LOVE" : expectedWord;
            bool strict = kind == GuidedQuestOutcomeKind.StrictSuccess;
            PersonalLiveCaptureState state = kind == GuidedQuestOutcomeKind.TechnicalFailure
                ? PersonalLiveCaptureState.Failed
                : PersonalLiveCaptureState.Completed;
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                state,
                kind != GuidedQuestOutcomeKind.TechnicalFailure,
                expectedWord,
                predicted,
                strict,
                strict,
                strict,
                kind == GuidedQuestOutcomeKind.TechnicalFailure,
                false);
            return new PracticeRecognitionAttempt(
                Guid.NewGuid().ToString("N"),
                expectedWord,
                predicted,
                state,
                outcome,
                null,
                null,
                "test-route",
                "test-profile",
                0.5d,
                null,
                null);
        }

        static string Read(string relativePath)
        {
            string root = Directory.GetParent(Application.dataPath).FullName;
            return File.ReadAllText(Path.Combine(root,
                relativePath.Replace('/', Path.DirectorySeparatorChar)));
        }

        static string MethodBody(string source, string signature)
        {
            int signatureOffset = source.IndexOf(signature, StringComparison.Ordinal);
            Assert.GreaterOrEqual(signatureOffset, 0, signature);
            int openBrace = source.IndexOf('{', signatureOffset);
            int depth = 0;
            for (int index = openBrace; index < source.Length; index++)
            {
                if (source[index] == '{') depth++;
                else if (source[index] == '}' && --depth == 0)
                    return source.Substring(openBrace + 1, index - openBrace - 1);
            }
            Assert.Fail("method body is not closed: " + signature);
            return string.Empty;
        }

        static int Count(string text, string token)
        {
            int count = 0;
            int offset = 0;
            while ((offset = text.IndexOf(token, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += token.Length;
            }
            return count;
        }

        static void InvokePrivate(object target, string name)
        {
            MethodInfo method = target.GetType().GetMethod(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(method, name);
            method.Invoke(target, null);
        }

        static T GetField<T>(object target, string name) where T : class
        {
            FieldInfo field = target.GetType().GetField(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(field, name);
            return field.GetValue(target) as T;
        }

        static T GetValue<T>(object target, string name)
        {
            FieldInfo field = target.GetType().GetField(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(field, name);
            return (T)field.GetValue(target);
        }

        static void SetField(object target, string name, object value)
        {
            FieldInfo field = target.GetType().GetField(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(field, name);
            field.SetValue(target, value);
        }

        static int EventSubscriberCount(object target, string name)
        {
            FieldInfo field = target.GetType().GetField(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(field, name);
            Delegate value = field.GetValue(target) as Delegate;
            return value == null ? 0 : value.GetInvocationList().Length;
        }

        sealed class WiringRig : IDisposable
        {
            GameObject controllerHost;

            public FakeAdapter Adapter { get; }
            public FakeRepository Repository { get; }
            public FakeObservationSource Source { get; }
            public FakeCamera Camera { get; }
            public RawMediaPipeNeutralAutoSegmenter Segmenter { get; }
            public PetRobotSignPracticeSession Session { get; }
            public PetRobotNeutralAutoCaptureCoordinator Coordinator { get; }
            public GuidedSignPracticePresentationController Controller { get; private set; }
            public MoaStoryGuidedPracticeFlow Flow =>
                GetField<MoaStoryGuidedPracticeFlow>(Controller, "storyFlow");

            public WiringRig(
                PersonalLiveCaptureState initialCaptureState = PersonalLiveCaptureState.Ready,
                SignDemonstrationAvatarPresenter stage = null)
            {
                Adapter = new FakeAdapter(initialCaptureState);
                Repository = new FakeRepository();
                Camera = new FakeCamera();
                Source = new FakeObservationSource();
                Session = new PetRobotSignPracticeSession(Adapter, Repository, null, Camera);
                Segmenter = new RawMediaPipeNeutralAutoSegmenter("step10k_test");
                var service = new PetRobotNeutralCalibrationProfileService(
                    "step10k_test", Path.Combine(Path.GetTempPath(), "WordsInSilence_step10k_test"));
                Coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                    Session,
                    service,
                    Segmenter,
                    new RawMediaPipeObservationRingBuffer(
                        PetRobotNeutralAutoSegmentationSettings.Default.PreRollSeconds),
                    Source,
                    Camera);
                controllerHost = new GameObject("Step10K_WiringTestController");
                Controller = controllerHost.AddComponent<GuidedSignPracticePresentationController>();
                InvokePrivate(Controller, "BuildUguiHierarchy");
                if (stage != null) SetField(Controller, "visualStage", stage);
                Controller.Bind(Session, null, null, null, Coordinator);
            }

            public void ReachWordIntroduction()
            {
                Assert.IsTrue(Flow.AdvanceStory());
                Assert.IsTrue(Flow.AdvanceStory());
                Assert.IsTrue(Flow.AdvanceStory());
                InvokePrivate(Controller, "RefreshUgui");
                Assert.AreEqual(MoaStoryBeat.WordIntroduction, Flow.Beat);
            }

            public void DestroyController()
            {
                if (controllerHost == null) return;
                UnityEngine.Object.DestroyImmediate(controllerHost);
                controllerHost = null;
                Controller = null;
            }

            public void Dispose()
            {
                DestroyController();
                Coordinator.Dispose();
                Session.Dispose();
            }
        }

        sealed class StageRig : IDisposable
        {
            readonly GameObject host;
            readonly List<UnityEngine.Object> templates = new List<UnityEngine.Object>();

            public SignDemonstrationAvatarPresenter Presenter { get; }

            public StageRig()
            {
                host = new GameObject("Step10K_DemonstrationStageTest");
                Presenter = host.AddComponent<SignDemonstrationAvatarPresenter>();
                Configure("greetingCharacter", "greetingClip", "GREETING");
                Configure("loveCharacter", "loveClip", "LOVE");
                Configure("thankYouCharacter", "thankYouClip", "THANK_YOU");
            }

            void Configure(string characterField, string clipField, string word)
            {
                var wrapper = new GameObject(word + "_Wrapper");
                var child = new GameObject(word + "_Character");
                child.transform.SetParent(wrapper.transform, false);
                child.AddComponent<Animator>();
                var clip = new AnimationClip { name = word + "_Clip" };
                clip.SetCurve(string.Empty, typeof(Transform), "localPosition.x",
                    AnimationCurve.Linear(0f, 0f, 1f, 1f));
                templates.Add(wrapper);
                templates.Add(clip);
                SetField(Presenter, characterField, wrapper);
                SetField(Presenter, clipField, clip);
            }

            public void Dispose()
            {
                UnityEngine.Object.DestroyImmediate(host);
                foreach (UnityEngine.Object value in templates)
                    if (value != null) UnityEngine.Object.DestroyImmediate(value);
            }
        }

        sealed class FakeAdapter : ISegmentedPersonalLivePracticeRecognitionAdapter
        {
            public PersonalLiveCaptureState CaptureState { get; private set; }
            public string CaptureSessionId { get; private set; } = string.Empty;
            public string ActiveProfileId => "test-profile";
            public bool CandidateRouteActive => true;
            public int BeginCalls { get; private set; }
            public int ResetCalls { get; private set; }
            public PracticeRecognitionAttempt NextAttempt { get; set; }

            public FakeAdapter(PersonalLiveCaptureState state)
            {
                CaptureState = state;
            }

            public bool SetExpectedWord(string word)
            {
                return PersonalLiveRecognitionController.IsSupportedWord(word);
            }

            public bool BeginCapture()
            {
                BeginCalls++;
                if (CaptureState != PersonalLiveCaptureState.Ready) return false;
                CaptureState = PersonalLiveCaptureState.Capturing;
                CaptureSessionId = Guid.NewGuid().ToString("N");
                return true;
            }

            public bool BeginCaptureWithPreRoll(IReadOnlyList<RawMediaPipeObservation> preRoll)
            {
                return preRoll != null && BeginCapture();
            }

            public PracticeRecognitionAttempt FinalizeCapture()
            {
                PracticeRecognitionAttempt result = NextAttempt;
                if (result != null) CaptureState = result.CaptureState;
                return result;
            }

            public PracticeRecognitionAttempt FinalizeCaptureWithTrailingTrim(double trailingTrimSeconds)
            {
                return FinalizeCapture();
            }

            public PracticeRecognitionAttempt FailActiveCaptureFromSource(Exception exception)
            {
                return FinalizeCapture();
            }

            public bool ResetCapture()
            {
                ResetCalls++;
                CaptureState = PersonalLiveCaptureState.Ready;
                CaptureSessionId = string.Empty;
                return true;
            }
        }

        sealed class FakeRepository : IPersonalizationCandidateRepository
        {
            public string RootDirectory => string.Empty;
            public string LearnerId => "step10k_test";
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;
            public int SaveCalls { get; private set; }
            public CandidateCountSnapshot ReloadAndVerify() => Counts;

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                return new CandidateSaveResult(
                    CandidateStorageStatus.Saved, string.Empty, "test", Counts);
            }
        }

        sealed class FakeObservationSource : IPersonalLiveObservationSource
        {
            public bool IsBound => true;
            public event Action<RawMediaPipeObservation> ObservationArrived;
            public event Action<Exception> SourceFailed;

            public void EmitFailure(Exception exception)
            {
                SourceFailed?.Invoke(exception);
            }
        }

        sealed class FakeCamera : IPetRobotCandidateCameraReadiness, IPetRobotNeutralCameraContext
        {
            public event Action Changed { add { } remove { } }
            public CandidateCameraStartupState CameraState => CandidateCameraStartupState.Ready;
            public bool IsReady => true;
            public bool CameraStarted => true;
            public bool HasReceivedFrame => true;
            public bool ProviderObservationReady => true;
            public int CameraFrameCount => 1;
            public int ProviderObservationCount => 1;
            public string LastErrorCode => string.Empty;
            public string LastErrorMessage => string.Empty;

            public bool TrySnapshotCameraFingerprint(out CameraFingerprint fingerprint)
            {
                fingerprint = null;
                return false;
            }
        }
    }
}
