using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Calibration;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotNeutralAutoSegmentationV1Tests
    {
        const string LearnerId = "local_learner";
        string tempRoot;

        [SetUp]
        public void SetUp()
        {
            tempRoot = Path.Combine(Path.GetTempPath(), "wis_step10j_" + Guid.NewGuid().ToString("N"));
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(tempRoot)) Directory.Delete(tempRoot, true);
        }

        [Test]
        public void Settings_AreTheSingleApprovedInteractionSurface()
        {
            PetRobotNeutralAutoSegmentationSettings value = PetRobotNeutralAutoSegmentationSettings.Default;
            Assert.AreEqual(2.5d, value.NeutralCalibrationHoldSeconds);
            Assert.AreEqual(1.5d, value.ReadyNeutralHoldSeconds);
            Assert.AreEqual(0.03d, value.OnsetDisplacementThreshold);
            Assert.AreEqual(3, value.OnsetConsecutiveFrames);
            Assert.AreEqual(0.35d, value.PreRollSeconds);
            Assert.AreEqual(0.5d, value.CaptureGraceSeconds);
            Assert.AreEqual(6d, value.ArmedTimeoutSeconds);
            Assert.AreEqual(10d, value.MaxCaptureSeconds);
            Assert.AreEqual(1d, value.EndNeutralHoldSeconds);
            Assert.AreEqual(0.25d, value.PostCaptureMarginSeconds);
            Assert.AreEqual(0.75d, value.AutoTrailingTrimSeconds);
            Assert.AreEqual(45f, value.NeutralArmsDownMaxDegrees);
        }

        [Test]
        public void MissingProfile_FirstNeutralCalibrationRequiresTwoPointFiveSeconds()
        {
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.RequireCalibration(Camera(), "missing");
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration, segmenter.State);
            Assert.IsTrue(segmenter.RequestAttempt());
            Assert.IsNull(segmenter.Process(Neutral(0d), true).CalibrationProfile);
            Assert.IsNull(segmenter.Process(Neutral(2.49d), true).CalibrationProfile);
            UserCalibrationProfile result = segmenter.Process(Neutral(2.5d), true).CalibrationProfile;
            Assert.IsNotNull(result);
            Assert.AreEqual(LearnerId, result.UserProfileId);
            Assert.AreEqual(0.2f, result.BodyScale.ShoulderWidthNorm, 0.0001f);
            Assert.IsTrue(result.Neutral.BottomExitAllowed);
        }

        [Test]
        public void FirstCalibration_ExplicitRequest_AllowsBothPoseWristsUnavailableWhenArmsDownAndHandsAbsent()
        {
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.RequireCalibration(Camera(), "missing");
            Assert.IsTrue(segmenter.RequestAttempt());
            Assert.IsNull(segmenter.Process(Neutral(0d, poseWrists: false), true).CalibrationProfile);
            UserCalibrationProfile result =
                segmenter.Process(Neutral(2.5d, poseWrists: false), true).CalibrationProfile;
            Assert.IsNotNull(result);
            Assert.IsFalse(result.Neutral.HandsVisible);
            Assert.IsTrue(result.Neutral.BottomExitAllowed);
            Assert.IsFalse(result.Neutral.HasWristBaseline);
        }

        [Test]
        public void FirstCalibration_NoRequest_DoesNotAdvanceOrEmitProfile()
        {
            PetRobotNeutralCalibrationProfileService service = Service();
            var source = new FakeObservationSource();
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                new PetRobotSignPracticeSession(new FakeSegmentedAdapter(), new FakeRepository()),
                service, segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext()))
            {
                foreach (double timestamp in new[] { 0d, 1d, 2d, 3d })
                    source.Emit(Neutral(timestamp, poseWrists: false));
                Assert.AreEqual(0f, coordinator.CalibrationProgress);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration, coordinator.State);
                Assert.IsFalse(File.Exists(service.ProfilePath));
            }
        }

        [TestCase("side")]
        [TestCase("top")]
        public void FirstCalibration_ValidSideOrTopWristEvidenceRejects(string mode)
        {
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.RequireCalibration(Camera(), "missing");
            Assert.IsTrue(segmenter.RequestAttempt());
            RawMediaPipeNeutralAutoSegmenterDecision decision = segmenter.Process(
                Neutral(0d,
                    wristMode: mode == "side" ? WristMode.Side : WristMode.Top,
                    leftPoseWrist: true,
                    rightPoseWrist: false), true);
            Assert.IsNull(decision.CalibrationProfile);
            Assert.AreEqual(NeutralObservationReason.WristNotBottomCompatible, segmenter.LastNeutralReason);
            Assert.AreEqual(0f, segmenter.CalibrationProgress);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration, segmenter.State);
        }

        [Test]
        public void FirstCalibration_OneBottomWristAndOppositeUnavailable_IsAllowedWhenExplicitlyGuided()
        {
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.RequireCalibration(Camera(), "missing");
            Assert.IsTrue(segmenter.RequestAttempt());
            Assert.IsNull(segmenter.Process(
                Neutral(0d, leftPoseWrist: true, rightPoseWrist: false), true).CalibrationProfile);
            UserCalibrationProfile result = segmenter.Process(
                Neutral(2.5d, leftPoseWrist: true, rightPoseWrist: false), true).CalibrationProfile;
            Assert.IsNotNull(result);
            Assert.IsFalse(result.Neutral.HandsVisible);
            Assert.IsTrue(result.Neutral.BottomExitAllowed);
            Assert.IsFalse(result.Neutral.HasWristBaseline);
        }

        [Test]
        public void ExplicitRecalibrationRequest_StartsCalibrationCollection()
        {
            var segmenter = SavedSegmenter();
            segmenter.RequestRecalibration(Camera());
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration, segmenter.State);
            Assert.IsNotEmpty(segmenter.AttemptId);
            Assert.IsNull(segmenter.Process(Neutral(0d, poseWrists: false), true).CalibrationProfile);
            UserCalibrationProfile result =
                segmenter.Process(Neutral(2.5d, poseWrists: false), true).CalibrationProfile;
            Assert.IsNotNull(result);
            Assert.AreEqual(LearnerId, result.UserProfileId);
        }

        [Test]
        public void Profile_NormalSaveLoadAndSignIndependentReuse()
        {
            var service = Service();
            UserCalibrationProfile profile = Profile(Camera(), 0.2f);
            Assert.AreEqual(NeutralCalibrationProfileSaveStatus.Saved, service.Save(profile, false).Status);
            NeutralCalibrationProfileLoadResult loaded = service.Load(Camera());
            Assert.AreEqual(NeutralCalibrationProfileLoadStatus.Valid, loaded.Status);
            Assert.AreEqual(LearnerId, loaded.Profile.UserProfileId);
            StringAssert.DoesNotContain("GREETING", service.ProfilePath);
            StringAssert.DoesNotContain("LOVE", service.ProfilePath);
            StringAssert.DoesNotContain("THANK_YOU", service.ProfilePath);
        }

        [Test]
        public void Profile_NextWordReuseDoesNotRequestFullCalibration()
        {
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.UseSavedProfile(Profile(Camera(), 0.2f), Camera());
            Assert.IsTrue(segmenter.RequestAttempt());
            segmenter.Process(Neutral(0d), true);
            segmenter.Process(Neutral(1.5d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, segmenter.State);
            segmenter.Cancel();
            Assert.IsTrue(segmenter.RequestAttempt());
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral, segmenter.State);
            Assert.IsTrue(segmenter.IsProfileLoaded);
        }

        [Test]
        public void Profile_CorruptJsonFailsSafeToNeedsCalibration()
        {
            Directory.CreateDirectory(tempRoot);
            File.WriteAllText(Path.Combine(tempRoot, LearnerId + ".json"), "{ definitely not json");
            NeutralCalibrationProfileLoadResult result = Service().Load(Camera());
            Assert.AreEqual(NeutralCalibrationProfileLoadStatus.Corrupt, result.Status);
            Assert.IsTrue(result.NeedsCalibration);
            Assert.IsNull(result.Profile);
        }

        [Test]
        public void Profile_CameraAndMirrorMismatchFailSafe()
        {
            var service = Service();
            Assert.IsTrue(service.Save(Profile(Camera(), 0.2f), false).IsSaved);
            CameraFingerprint changed = Camera();
            changed.PreviewMirrored = !changed.PreviewMirrored;
            Assert.AreEqual(NeutralCalibrationProfileLoadStatus.CameraMismatch, service.Load(changed).Status);
            changed = Camera();
            changed.RotationDegrees = 90;
            Assert.AreEqual(NeutralCalibrationProfileLoadStatus.CameraMismatch, service.Load(changed).Status);
        }

        [Test]
        public void SavedProfile_ClearBodyScaleMismatchReturnsToCalibration()
        {
            var segmenter = SavedSegmenter();
            segmenter.RequestAttempt();
            segmenter.Process(Neutral(0d, shoulderWidth: 0.35f), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration, segmenter.State);
            Assert.AreEqual(NeutralObservationReason.BodyScaleMismatch, segmenter.LastNeutralReason);
            Assert.IsFalse(segmenter.IsProfileLoaded);
        }

        [Test]
        public void HandsMissingAlone_NeverMakesNonDownArmsNeutral()
        {
            var segmenter = SavedSegmenter();
            segmenter.RequestAttempt();
            segmenter.Process(Neutral(0d, armsDown: false, poseWrists: false), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral, segmenter.State);
            Assert.AreEqual(NeutralObservationReason.ArmsNotDown, segmenter.LastNeutralReason);
            Assert.AreEqual(0f, segmenter.NeutralHoldProgress);
        }

        [Test]
        public void ArmsDownBottomCompatibleNeutral_ArmsAfterOnePointFiveSecondHold()
        {
            var segmenter = SavedSegmenter();
            segmenter.RequestAttempt();
            segmenter.Process(Neutral(0d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral, segmenter.State);
            segmenter.Process(Neutral(1.5d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, segmenter.State);
            Assert.AreEqual(NeutralObservationReason.Neutral, segmenter.LastNeutralReason);
        }

        [TestCase("side")]
        [TestCase("top")]
        [TestCase("ambiguous")]
        public void SideTopAndAmbiguousWristLoss_AreRejected(string mode)
        {
            var segmenter = mode == "ambiguous"
                ? SavedHandsVisibleSegmenter()
                : SavedSegmenter();
            segmenter.RequestAttempt();
            RawMediaPipeObservation observation = mode == "side"
                ? Neutral(0d, wristMode: WristMode.Side)
                : mode == "top"
                    ? Neutral(0d, wristMode: WristMode.Top)
                    : Neutral(0d, poseWrists: false);
            segmenter.Process(observation, true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral, segmenter.State);
            Assert.AreEqual(
                mode == "ambiguous"
                    ? NeutralObservationReason.AmbiguousWristLoss
                    : NeutralObservationReason.WristNotBottomCompatible,
                segmenter.LastNeutralReason);
        }

        [Test]
        public void NonFiniteValidLandmark_IsRejectedBeforeNeutralLogic()
        {
            double[] coordinates = NaNs(RawMediaPipeObservation.PoseLandmarkCount);
            bool[] valid = new bool[RawMediaPipeObservation.PoseLandmarkCount];
            Write(coordinates, valid, 11, double.NaN, 0.3d, 0d);
            Assert.Throws<ArgumentException>(() => new RawLandmarkObservationSet(
                RawMediaPipeObservation.PoseLandmarkCount, coordinates, valid,
                new bool[RawMediaPipeObservation.PoseLandmarkCount],
                new bool[RawMediaPipeObservation.PoseLandmarkCount]));
        }

        [Test]
        public void Armed_StillNeutralAndLargeBodyRelativeJitter_DoesNotStart()
        {
            var segmenter = ArmedSegmenter();
            var timestamps = new[] { 1.6d, 1.7d, 1.8d, 1.9d, 2.0d, 2.1d, 2.2d };
            var offsets = new[] { 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f };

            for (int index = 0; index < timestamps.Length; index++)
            {
                RawMediaPipeNeutralAutoSegmenterDecision decision = segmenter.Process(
                    Neutral(timestamps[index], wristOffset: offsets[index]), true);
                Assert.IsFalse(decision.BeginCaptureRequested);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, segmenter.State);
            }
        }

        [Test]
        public void Armed_ShortHandFlicker_DoesNotStart()
        {
            var segmenter = ArmedSegmenter();
            Assert.IsFalse(segmenter.Process(Neutral(1.6d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(Neutral(1.65d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(Neutral(1.7d), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(Neutral(1.8d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(Neutral(1.99d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(Neutral(2.0d), true).BeginCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, segmenter.State);
        }

        [Test]
        public void RawRing_IsTimestampBoundedOldestFirstAndKeepsCurrentOnsetOnce()
        {
            var ring = new RawMediaPipeObservationRingBuffer(0.35d);
            RawMediaPipeObservation onset = Neutral(1.4d, wristOffset: 0.03f);
            ring.Add(Neutral(1.0d));
            ring.Add(Neutral(1.2d));
            ring.Add(onset);
            IReadOnlyList<RawMediaPipeObservation> snapshot = ring.Snapshot();
            CollectionAssert.AreEqual(new[] { 1.2d, 1.4d }, snapshot.Select(value => value.TimestampSeconds));
            Assert.AreEqual(1, snapshot.Count(value => ReferenceEquals(value, onset)));
        }

        [Test]
        public void RawRing_RejectsNullDuplicateAndOutOfOrderTimestamps()
        {
            var ring = new RawMediaPipeObservationRingBuffer(0.35d);
            Assert.Throws<ArgumentNullException>(() => ring.Add(null));
            ring.Add(Neutral(1d));
            Assert.Throws<ArgumentException>(() => ring.Add(Neutral(1d)));
            Assert.Throws<ArgumentException>(() => ring.Add(Neutral(0.9d)));
        }

        [Test]
        public void CurrentController_PreRollPrimesRawCaptureAndOnsetIsNotDuplicated()
        {
            FakeObservationSource source;
            PersonalLiveRecognitionController controller = ReadyController(out source);
            RawMediaPipeObservation onset = RecognitionObservation(1.2d);
            var preRoll = new[] { RecognitionObservation(1d), onset };
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(preRoll, "auto-session"));
            Assert.AreEqual(2, controller.BufferedRawFrameCount);
            Assert.AreEqual("auto-session", controller.CaptureSessionId);
            source.Emit(onset);
            Assert.AreEqual(2, controller.BufferedRawFrameCount);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            source.Emit(RecognitionObservation(1.3d));
            Assert.AreEqual(3, controller.BufferedRawFrameCount);
            controller.Dispose();
        }

        [Test]
        public void CurrentController_ManualBeginCaptureRemainsEmptyAndUnprimed()
        {
            FakeObservationSource source;
            PersonalLiveRecognitionController controller = ReadyController(out source);
            Assert.IsTrue(controller.BeginCapture("manual-session"));
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
            source.Emit(RecognitionObservation(1d));
            Assert.AreEqual(1, controller.BufferedRawFrameCount);
            controller.Dispose();
        }

        [Test]
        public void CurrentController_AutoTrimDropsPointSevenFiveSecondsAndKeepsEarlierFrames()
        {
            FakeObservationSource source;
            PersonalLiveRecognitionController controller = ReadyController(out source);
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(
                Enumerable.Range(0, 9).Select(index => RecognitionObservation(index * 0.25d)).ToArray()));
            Assert.IsTrue(controller.FinalizeCaptureWithTrailingTrim(0.75d));
            Assert.AreEqual(6, controller.Result.RawFrameCount);
            controller.Dispose();
        }

        [Test]
        public void CurrentController_ManualFinalizeUsesZeroTrim()
        {
            FakeObservationSource source;
            PersonalLiveRecognitionController controller = ReadyController(out source);
            Assert.IsTrue(controller.BeginCapture());
            source.Emit(RecognitionObservation(0d));
            source.Emit(RecognitionObservation(0.25d));
            Assert.IsTrue(controller.FinalizeCapture());
            Assert.AreEqual(2, controller.Result.RawFrameCount);
            controller.Dispose();
        }

        [Test]
        public void Controller_FailActiveCaptureFromSource_CapturingBecomesTypedBridgeFailure()
        {
            FakeObservationSource source;
            PersonalLiveRecognitionController controller = ReadyController(out source);
            var failure = new InvalidOperationException("fatal active bridge failure");
            Assert.IsTrue(controller.BeginCapture("fatal-controller-session"));

            Assert.IsTrue(controller.FailActiveCaptureFromSource(failure));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.IsNull(controller.Result);
            Assert.AreEqual("fatal-controller-session", controller.CaptureSessionId);
            Assert.IsNotNull(controller.Failure);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            Assert.AreEqual(typeof(InvalidOperationException).FullName, controller.Failure.ExceptionType);
            StringAssert.AreEqualIgnoringCase(
                "Step 9C observation bridge failed: fatal active bridge failure",
                controller.Failure.Message);
            Assert.IsNotNull(controller.PrimaryFailure);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing,
                controller.PrimaryFailure.StateBeforeFailure);
            Assert.AreEqual("fatal-controller-session", controller.PrimaryFailure.CaptureSessionId);
            controller.Dispose();
        }

        [Test]
        public void Controller_FailActiveCaptureFromSource_AlreadyFailedIsIdempotentAndPreservesPrimaryFailure()
        {
            FakeObservationSource source;
            PersonalLiveRecognitionController controller = ReadyController(out source);
            Assert.IsTrue(controller.BeginCapture("primary-failure-session"));
            Assert.IsTrue(controller.FailActiveCaptureFromSource(
                new InvalidOperationException("first failure")));
            PersonalLiveFailure failure = controller.Failure;
            PersonalLivePrimaryFailureSnapshot primary = controller.PrimaryFailure;

            Assert.IsTrue(controller.FailActiveCaptureFromSource(
                new ArgumentException("late duplicate failure")));

            Assert.AreSame(failure, controller.Failure);
            Assert.AreSame(primary, controller.PrimaryFailure);
            Assert.AreEqual("first failure", primary.Message.Substring(
                primary.Message.Length - "first failure".Length));
            Assert.AreEqual("primary-failure-session", controller.CaptureSessionId);
            controller.Dispose();
        }

        [Test]
        public void Controller_FailActiveCaptureFromSource_NonActiveStateDoesNotMutate()
        {
            var idle = new PersonalLiveRecognitionController(
                new FakeObservationSource(), new FakeRuntimeFactory(new FakeRuntime(SuccessRouting())));
            Assert.IsFalse(idle.FailActiveCaptureFromSource(new InvalidOperationException("idle")));
            Assert.AreEqual(PersonalLiveCaptureState.Idle, idle.State);
            Assert.IsNull(idle.Failure);
            idle.Dispose();

            FakeObservationSource readySource;
            PersonalLiveRecognitionController ready = ReadyController(out readySource);
            Assert.IsFalse(ready.FailActiveCaptureFromSource(new InvalidOperationException("ready")));
            Assert.AreEqual(PersonalLiveCaptureState.Ready, ready.State);
            Assert.IsNull(ready.Failure);
            ready.Dispose();

            FakeObservationSource completedSource;
            PersonalLiveRecognitionController completed = ReadyController(out completedSource);
            Assert.IsTrue(completed.BeginCapture());
            completedSource.Emit(RecognitionObservation(0d));
            Assert.IsTrue(completed.FinalizeCapture());
            Assert.IsFalse(completed.FailActiveCaptureFromSource(
                new InvalidOperationException("completed")));
            Assert.AreEqual(PersonalLiveCaptureState.Completed, completed.State);
            completed.Dispose();

            FakeObservationSource cancelledSource;
            PersonalLiveRecognitionController cancelled = ReadyController(out cancelledSource);
            Assert.IsTrue(cancelled.BeginCapture());
            Assert.IsTrue(cancelled.CancelCapture());
            Assert.IsFalse(cancelled.FailActiveCaptureFromSource(
                new InvalidOperationException("cancelled")));
            Assert.AreEqual(PersonalLiveCaptureState.Cancelled, cancelled.State);
            cancelled.Dispose();
        }

        [Test]
        public void Session_AutoTechnicalFailure_UsesExistingFinishCoreAndEntersFeedback()
        {
            var adapter = new FakeSegmentedAdapter
            {
                FailureNext = Attempt(GuidedQuestOutcomeKind.TechnicalFailure),
            };
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            Assert.IsTrue(session.StartAutoExpression(new[] { Neutral(0d) }));

            Assert.IsTrue(session.FinishAutoTechnicalFailure(
                new InvalidOperationException("fatal auto source")));

            Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
            Assert.IsNotNull(session.LastLearningOutcome);
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                session.LastLearningOutcome.Kind);
            Assert.IsTrue(session.CanRetry);
            session.Dispose();
        }

        [Test]
        public void Session_AutoTechnicalFailure_WritesNoCandidate()
        {
            var adapter = new FakeSegmentedAdapter
            {
                FailureNext = Attempt(GuidedQuestOutcomeKind.TechnicalFailure),
            };
            var repository = new FakeRepository();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            Assert.IsTrue(session.StartAutoExpression(new[] { Neutral(0d) }));

            Assert.IsTrue(session.FinishAutoTechnicalFailure(
                new InvalidOperationException("fatal auto source")));

            Assert.AreEqual(0, repository.SaveCalls);
            Assert.AreEqual(CandidateStorageStatus.NotRequested,
                session.LastStorageResult.Status);
            Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                session.GetWordLearningStatus("GREETING"));
            session.Dispose();
        }

        [Test]
        public void Session_AutoTechnicalFailure_PreservesSessionIdExpectedWordAndTechnicalOutcome()
        {
            const string captureId = "recovered-capture-id";
            var adapter = new FakeSegmentedAdapter
            {
                FailureNext = Attempt(
                    GuidedQuestOutcomeKind.TechnicalFailure, captureId, "GREETING"),
            };
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            Assert.IsTrue(session.StartAutoExpression(new[] { Neutral(0d) }));
            Assert.IsTrue(session.FinishAutoTechnicalFailure(
                new InvalidOperationException("fatal auto source")));

            Assert.AreEqual(captureId, session.LastAttempt.CaptureSessionId);
            Assert.AreEqual("GREETING", session.LastAttempt.ExpectedWord);
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                session.LastAttempt.Outcome.Kind);
            Assert.AreEqual(PersonalLiveCaptureState.Failed,
                session.LastAttempt.CaptureState);
            session.Dispose();
        }

        [Test]
        public void Session_AutoTechnicalFailure_DoesNotCallManualOrNormalAutoFinalize()
        {
            var adapter = new FakeSegmentedAdapter
            {
                FailureNext = Attempt(GuidedQuestOutcomeKind.TechnicalFailure),
            };
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            Assert.IsTrue(session.StartAutoExpression(new[] { Neutral(0d) }));

            Assert.IsTrue(session.FinishAutoTechnicalFailure(
                new InvalidOperationException("fatal auto source")));

            Assert.AreEqual(1, adapter.FailureRecoveryCalls);
            Assert.AreEqual(0, adapter.ManualFinishCalls);
            Assert.AreEqual(0, adapter.AutoFinishCalls);
            session.Dispose();
        }

        [Test]
        public void Session_RetryAfterAutoTechnicalFailure_ReturnsReady()
        {
            var adapter = new FakeSegmentedAdapter
            {
                FailureNext = Attempt(GuidedQuestOutcomeKind.TechnicalFailure),
            };
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            Assert.IsTrue(session.StartAutoExpression(new[] { Neutral(0d) }));
            Assert.IsTrue(session.FinishAutoTechnicalFailure(
                new InvalidOperationException("fatal auto source")));

            Assert.IsTrue(session.Retry());

            Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
            Assert.AreEqual(PersonalLiveCaptureState.Ready, adapter.CaptureState);
            Assert.AreEqual(1, adapter.ResetCalls);
            session.Dispose();
        }

        [Test]
        public void Coordinator_GenericSourceFailureDuringActiveCapture_RecoversSessionToFeedback()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(new InvalidOperationException("generic active failure"));

                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
                Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                    session.LastLearningOutcome.Kind);
                Assert.AreEqual(1, adapter.FailureRecoveryCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_FirstActiveAmbiguity_RemainsNonFatal()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(Ambiguity(1));

                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, session.State);
                Assert.AreEqual(0, adapter.FailureRecoveryCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_SecondActiveAmbiguity_RecoversAsTechnicalFailure()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(Ambiguity(1));
                source.EmitFailure(Ambiguity(2));

                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
                Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                    session.LastLearningOutcome.Kind);
                Assert.AreEqual(1, adapter.FailureRecoveryCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_LateDuplicateFailureAfterRecovery_DoesNotDoubleComplete()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(new InvalidOperationException("first fatal"));
                PracticeRecognitionAttempt attempt = session.LastAttempt;
                source.EmitFailure(new InvalidOperationException("late duplicate"));

                Assert.AreSame(attempt, session.LastAttempt);
                Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
                Assert.AreEqual(1, adapter.FailureRecoveryCalls);
                Assert.AreEqual(0, adapter.AutoFinishCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_RetryThenNewRequestAttempt_ReentersNeutralFlow()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(new InvalidOperationException("fatal before retry"));
                Assert.IsTrue(session.Retry());

                DriveCoordinatorToCapturing(coordinator, source, 4d);
                source.EmitFailure(Ambiguity(3));

                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, session.State);
                Assert.AreEqual(1, adapter.FailureRecoveryCalls);
                Assert.AreEqual(1, adapter.ResetCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_PreCaptureGenericFailure_DoesNotFabricateSessionOutcome()
        {
            var source = new FakeObservationSource();
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            var adapter = new FakeSegmentedAdapter();
            var repository = new FakeRepository();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                session, Service(), segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext()))
            {
                Assert.IsTrue(coordinator.RequestAttempt());
                source.EmitFailure(new InvalidOperationException("pre-capture generic"));

                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
                Assert.IsNull(session.LastAttempt);
                Assert.IsNull(session.LastLearningOutcome);
                Assert.AreEqual(0, repository.SaveCalls);
                Assert.AreEqual(0, adapter.FailureRecoveryCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_EventOrder_ControllerFirst_ProducesSingleTechnicalFailureFeedback()
        {
            using (var rig = new RecoveryRig(Service()))
            {
                DriveCoordinatorToCapturing(rig.Coordinator, rig.Source, 0d);
                rig.Source.EmitFailure(new InvalidOperationException("controller-first fatal"));

                AssertRecoveredTechnicalFailure(rig, "controller-first fatal");
            }
        }

        [Test]
        public void Coordinator_EventOrder_CoordinatorFirst_ProducesSingleTechnicalFailureFeedback()
        {
            using (var rig = new RecoveryRig(Service()))
            {
                DriveCoordinatorToCapturing(rig.Coordinator, rig.Source, 0d);
                rig.Source.EmitFailureReverse(new InvalidOperationException("coordinator-first fatal"));

                AssertRecoveredTechnicalFailure(rig, "coordinator-first fatal");
            }
        }

        [Test]
        public void EndCandidate_ResumedMotionKeepsAttemptAndCaptureState()
        {
            var segmenter = CapturingSegmenter();
            string attempt = segmenter.AttemptId;
            segmenter.Process(Neutral(2.4d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.EndCandidate, segmenter.State);
            // F32 cadence: the next OBSERVED frame, not a skipped 0.4 s leap
            segmenter.Process(Neutral(2.4d + FrameStep, armsDown: false), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
            Assert.AreEqual(attempt, segmenter.AttemptId);
            Assert.AreEqual(0f, segmenter.EndHoldProgress);
            Assert.AreEqual(NeutralEndpointHoldReset.NeutralLost, segmenter.LastEndpointHoldReset);
        }

        [Test]
        public void AutoEnd_OneSecondOfObservedNeutralRequestsFinalizeExactlyOnceWithApprovedTrim()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.EndCandidate, segmenter.State);
            // 23 observed frames = 0.989 s: still short of the intended margin
            HoldNeutralWithoutFinalize(segmenter, 2.4d, 23);
            RawMediaPipeNeutralAutoSegmenterDecision decision =
                segmenter.Process(Neutral(2.4d + 24 * FrameStep), true);
            Assert.IsTrue(decision.FinalizeCaptureRequested);
            Assert.AreEqual(0.75d, decision.TrailingTrimSeconds);
            Assert.AreEqual(NeutralEndpointResolution.StableNeutralHold,
                segmenter.LastEndpointResolution);
            Assert.IsFalse(segmenter.Process(
                Neutral(2.4d + 25 * FrameStep), true).FinalizeCaptureRequested);
        }

        [Test]
        public void EndCandidate_UnobservedGapResetsTheHoldAndTheNextHoldIsNotTruncated()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            HoldNeutralWithoutFinalize(segmenter, 2.4d, 20);   // 0.86 s observed

            // one skipped second: the learner was NOT observed here
            Assert.IsFalse(segmenter.Process(Neutral(3.5d), true).FinalizeCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
            Assert.AreEqual(NeutralEndpointHoldReset.ObservationGap,
                segmenter.LastEndpointHoldReset);
            Assert.AreEqual(0f, segmenter.EndHoldProgress);

            segmenter.Process(Neutral(3.5d + FrameStep), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.EndCandidate, segmenter.State);
            HoldNeutralWithoutFinalize(segmenter, 3.5d + FrameStep, 23);
            Assert.IsTrue(segmenter.Process(
                Neutral(3.5d + 25 * FrameStep), true).FinalizeCaptureRequested);
        }

        [Test]
        public void EndCandidate_CameraInterruptionResetsTheHoldInsteadOfCountingIt()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            HoldNeutralWithoutFinalize(segmenter, 2.4d, 20);

            Assert.IsFalse(segmenter.Process(
                Neutral(2.4d + 21 * FrameStep), false).FinalizeCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
            Assert.AreEqual(NeutralObservationReason.CameraNotReady, segmenter.LastNeutralReason);
            Assert.AreEqual(NeutralEndpointHoldReset.CameraNotReady,
                segmenter.LastEndpointHoldReset);

            // a returning camera must be observed for a FULL second again
            double resume = 2.4d + 22 * FrameStep;
            segmenter.Process(Neutral(resume), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.EndCandidate, segmenter.State);
            HoldNeutralWithoutFinalize(segmenter, resume, 23);
            Assert.IsTrue(segmenter.Process(
                Neutral(resume + 24 * FrameStep), true).FinalizeCaptureRequested);
        }

        [Test]
        public void EndCandidate_NonIncreasingTimestampResetsTheHold()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            HoldNeutralWithoutFinalize(segmenter, 2.4d, 10);
            Assert.IsFalse(segmenter.Process(
                Neutral(2.4d + 5 * FrameStep), true).FinalizeCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
            Assert.AreEqual(NeutralEndpointHoldReset.NonIncreasingTimestamp,
                segmenter.LastEndpointHoldReset);
        }

        [Test]
        public void EndCandidate_NaturalNeutralRelativeJitterStillEndsAfterOneSecond()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            HoldNeutralWithoutFinalize(segmenter, 2.4d, 23,
                index => index % 2 == 0 ? 0.1f : -0.1f);
            RawMediaPipeNeutralAutoSegmenterDecision decision = segmenter.Process(
                Neutral(2.4d + 24 * FrameStep, wristOffset: 0.1f), true);
            Assert.IsTrue(decision.FinalizeCaptureRequested);
            Assert.AreEqual(0.75d, decision.TrailingTrimSeconds, 1e-9d);
            Assert.AreEqual(NeutralEndpointResolution.StableNeutralHold,
                segmenter.LastEndpointResolution);
        }

        [Test]
        public void EndCandidate_NeutralDriftStillWaitsTheFullSecondThenEnds()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            // a steady body-relative drift that never leaves the saved neutral: it may not
            // finalize early, but it also may never reset the hold
            for (int index = 1; index <= 23; index++)
                Assert.IsFalse(segmenter.Process(
                    Neutral(2.4d + index * FrameStep, wristOffset: index * 0.003f), true)
                    .FinalizeCaptureRequested,
                    "a held neutral must not finalize before one second");
            RawMediaPipeNeutralAutoSegmenterDecision decision = segmenter.Process(
                Neutral(2.4d + 24 * FrameStep, wristOffset: 24 * 0.003f), true);
            Assert.IsTrue(decision.FinalizeCaptureRequested);
            Assert.AreEqual(0.75d, decision.TrailingTrimSeconds, 1e-9d);
            Assert.AreEqual(NeutralEndpointResolution.StableNeutralHold,
                segmenter.LastEndpointResolution);
        }

        [Test]
        public void EndCandidate_TemporaryWristLossAloneDoesNotResetTheBottomExitHold()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.4d), true);
            for (int index = 1; index <= 23; index++)
                Assert.IsFalse(segmenter.Process(
                    Neutral(2.4d + index * FrameStep,
                            poseWrists: index < 10 || index > 13), true)
                    .FinalizeCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.EndCandidate, segmenter.State);
            Assert.IsTrue(segmenter.Process(
                Neutral(2.4d + 24 * FrameStep), true).FinalizeCaptureRequested);
        }

        [Test]
        public void CaptureGracePreventsEarlyNeutralEndCandidate()
        {
            var segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(2.29d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
            segmenter.Process(Neutral(2.31d), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.EndCandidate, segmenter.State);
        }

        [Test]
        public void MaxCaptureTenSecondsFailsSafeToSingleEvaluation()
        {
            var segmenter = CapturingSegmenter();
            RawMediaPipeNeutralAutoSegmenterDecision decision =
                segmenter.Process(Neutral(11.8d, armsDown: false), true);
            Assert.IsTrue(decision.FinalizeCaptureRequested);
            Assert.IsTrue(decision.MaxCaptureReached);
            Assert.AreEqual(0d, decision.TrailingTrimSeconds);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Evaluating, segmenter.State);
            Assert.AreEqual(NeutralEndpointResolution.MaxCaptureTimeout,
                segmenter.LastEndpointResolution,
                "the terminal timeout must be reported as a timeout, not as a stable end");
        }

        [TestCase(GuidedQuestOutcomeKind.StrictSuccess, 1, CandidateStorageStatus.Saved,
            PersonalizationCandidateEligibility.Eligible)]
        [TestCase(GuidedQuestOutcomeKind.LowConfidenceCorrect, 1, CandidateStorageStatus.Saved,
            PersonalizationCandidateEligibility.ReviewOnly)]
        [TestCase(GuidedQuestOutcomeKind.WrongWord, 0, CandidateStorageStatus.NotRequested,
            PersonalizationCandidateEligibility.Eligible)]
        [TestCase(GuidedQuestOutcomeKind.TechnicalFailure, 0, CandidateStorageStatus.NotRequested,
            PersonalizationCandidateEligibility.Eligible)]
        public void AutoSession_ReusesAuthoritativeOutcomeCandidatePolicy(
            GuidedQuestOutcomeKind kind,
            int expectedSaves,
            CandidateStorageStatus storageStatus,
            PersonalizationCandidateEligibility expectedEligibility)
        {
            var adapter = new FakeSegmentedAdapter { Next = Attempt(kind) };
            var repository = new FakeRepository();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            Assert.IsTrue(session.StartAutoExpression(new[] { Neutral(0d) }));
            Assert.IsTrue(session.FinishAutoExpression(0.75d));
            Assert.AreEqual(expectedSaves, repository.SaveCalls);
            Assert.AreEqual(storageStatus, session.LastStorageResult.Status);
            if (expectedSaves > 0) Assert.AreEqual(expectedEligibility, repository.LastEligibility);
            Assert.AreEqual(1, adapter.AutoBeginCalls);
            Assert.AreEqual(1, adapter.AutoFinishCalls);
            Assert.AreEqual(0, adapter.ManualBeginCalls);
            Assert.AreEqual(0, adapter.ManualFinishCalls);
        }

        [Test]
        public void Coordinator_DisposeUnsubscribesAndDoesNotOwnRecognitionOrStorageTypes()
        {
            var source = new FakeObservationSource();
            var adapter = new FakeSegmentedAdapter();
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                session,
                Service(),
                new RawMediaPipeNeutralAutoSegmenter(LearnerId),
                new RawMediaPipeObservationRingBuffer(0.35d),
                source,
                new FakeCameraContext());
            Assert.AreEqual(2, source.SubscriberCount);
            coordinator.Dispose();
            Assert.AreEqual(0, source.SubscriberCount);

            string sourceText = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotNeutralAutoCaptureCoordinator.cs");
            StringAssert.DoesNotContain("RecognitionCore", sourceText);
            StringAssert.DoesNotContain("MaskAwareDtw", sourceText);
            StringAssert.DoesNotContain("IPersonalizationCandidateRepository", sourceText);
        }

        [Test]
        public void Coordinator_PreCaptureAmbiguity_IsNonFatal()
        {
            var source = new FakeObservationSource();
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                new PetRobotSignPracticeSession(new FakeSegmentedAdapter(), new FakeRepository()),
                Service(), segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext()))
            {
                Assert.IsTrue(coordinator.RequestAttempt());
                source.EmitFailure(Ambiguity(1));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration, coordinator.State);
            }
        }

        [Test]
        public void Coordinator_FirstCaptureAmbiguity_IsNonFatal()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(Ambiguity(1));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, session.State);
            }
        }

        [Test]
        public void Coordinator_SecondCaptureAmbiguity_Fails()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(Ambiguity(1));
                source.EmitFailure(Ambiguity(2));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, coordinator.State);
            }
        }

        [Test]
        public void Coordinator_GenericSourceFailure_FailsActiveAttempt()
        {
            var source = new FakeObservationSource();
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                new PetRobotSignPracticeSession(new FakeSegmentedAdapter(), new FakeRepository()),
                Service(), segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext()))
            {
                Assert.IsTrue(coordinator.RequestAttempt());
                source.EmitFailure(new InvalidOperationException("generic bridge failure"));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, coordinator.State);
            }
        }

        [Test]
        public void Coordinator_LateFailureAfterEvaluating_DoesNotOverwriteState()
        {
            var source = new FakeObservationSource();
            RawMediaPipeNeutralAutoSegmenter segmenter = CapturingSegmenter();
            segmenter.Process(Neutral(11.8d, armsDown: false), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Evaluating, segmenter.State);
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                new PetRobotSignPracticeSession(new FakeSegmentedAdapter(), new FakeRepository()),
                Service(), segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext()))
            {
                source.EmitFailure(new InvalidOperationException("late bridge failure"));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Evaluating, coordinator.State);
            }
        }

        [Test]
        public void Coordinator_NewAttempt_ResetsCaptureAmbiguityAllowance()
        {
            FakeObservationSource source;
            FakeSegmentedAdapter adapter;
            PetRobotSignPracticeSession session;
            RawMediaPipeNeutralAutoSegmenter segmenter;
            using (PetRobotNeutralAutoCaptureCoordinator coordinator = CreateSavedCoordinator(
                out source, out adapter, out session, out segmenter))
            {
                adapter.Next = Attempt(GuidedQuestOutcomeKind.WrongWord);
                DriveCoordinatorToCapturing(coordinator, source, 0d);
                source.EmitFailure(Ambiguity(1));
                // F32: a realistic observed cadence, because one skipped second is no
                // longer accepted as a continuously observed end hold
                source.Emit(Neutral(2.4d));
                for (int index = 1; index <= 24; index++)
                    source.Emit(Neutral(2.4d + index * FrameStep));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Evaluating, coordinator.State);
                Assert.IsTrue(session.Retry());

                DriveCoordinatorToCapturing(coordinator, source, 4d);
                source.EmitFailure(Ambiguity(2));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, session.State);
            }
        }

        [Test]
        public void NewProductionSources_HaveNoForbiddenLegacyRuntimeReferences()
        {
            string[] files =
            {
                "PetRobotNeutralCalibrationProfileService.cs",
                "RawMediaPipeObservationRingBuffer.cs",
                "RawMediaPipeNeutralAutoSegmenter.cs",
                "PetRobotNeutralAutoCaptureCoordinator.cs",
            };
            string[] forbidden =
            {
                "CalibrationController", "NeutralPoseDetector", "AttemptStateMachine",
                "MotionOnsetDetector", "CircularFrameBuffer", "SignPracticeController",
                "SignRecognizer", "SignFeatureExtractor", "HandPoseEvaluator", "SVMClassifier",
                "HandPoseRecordingRepository", "PersonalDataRepository", "SignGameBridge",
                "SignPracticeEvents",
            };
            foreach (string file in files)
            {
                string source = ReadProject(
                    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/" + file);
                foreach (string symbol in forbidden)
                    StringAssert.DoesNotContain(symbol, source, file + " references " + symbol);
            }
        }

        [Test]
        public void ProfileTransaction_NormalAndExplicitRecalibrationUpdateAreAtomic()
        {
            var service = Service();
            Assert.IsTrue(service.Save(Profile(Camera(), 0.2f), false).IsSaved);
            UserCalibrationProfile update = Profile(Camera(), 0.21f);
            Assert.IsTrue(service.Save(update, true).IsSaved);
            Assert.AreEqual(0.21f, service.Load(Camera()).Profile.BodyScale.ShoulderWidthNorm, 0.0001f);
            AssertNoTransactionResidue();
        }

        [TestCase(FaultPoint.TempWrite)]
        [TestCase(FaultPoint.TempValidation)]
        [TestCase(FaultPoint.Replace)]
        [TestCase(FaultPoint.PostValidation)]
        public void ProfileTransaction_FailureInjectionPreservesExistingValidProfile(FaultPoint point)
        {
            var system = new SystemNeutralCalibrationProfileFileSystem();
            var baseline = new PetRobotNeutralCalibrationProfileService(
                LearnerId, tempRoot, system, FixedClock);
            Assert.IsTrue(baseline.Save(Profile(Camera(), 0.2f), false).IsSaved);
            string before = File.ReadAllText(baseline.ProfilePath);

            var fault = new FaultFileSystem(point, baseline.ProfilePath);
            var service = new PetRobotNeutralCalibrationProfileService(
                LearnerId, tempRoot, fault, FixedClock);
            NeutralCalibrationProfileSaveResult result = service.Save(Profile(Camera(), 0.21f), true);
            Assert.AreEqual(NeutralCalibrationProfileSaveStatus.TransactionFailed, result.Status);
            Assert.AreEqual(before, File.ReadAllText(baseline.ProfilePath));
            Assert.AreEqual(NeutralCalibrationProfileLoadStatus.Valid, baseline.Load(Camera()).Status);
            AssertNoTransactionResidue();
        }

        [Test]
        public void ProfileTransaction_FirstSaveMoveFailureLeavesNoPartialFinal()
        {
            string finalPath = Path.Combine(tempRoot, LearnerId + ".json");
            var fault = new FaultFileSystem(FaultPoint.Move, finalPath);
            var service = new PetRobotNeutralCalibrationProfileService(
                LearnerId, tempRoot, fault, FixedClock);
            NeutralCalibrationProfileSaveResult result = service.Save(Profile(Camera(), 0.2f), false);
            Assert.AreEqual(NeutralCalibrationProfileSaveStatus.TransactionFailed, result.Status);
            Assert.IsFalse(File.Exists(finalPath));
            AssertNoTransactionResidue();
        }

        [Test]
        public void Session_PrepareAutoAttempt_WhenRecognitionReady_DoesNotReset()
        {
            var adapter = new FakeSegmentedAdapter();
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());

            Assert.IsTrue(session.PrepareAutoAttempt());

            Assert.AreEqual(0, adapter.ResetCalls);
            Assert.AreEqual("GREETING", adapter.LastExpectedWord);
            Assert.AreEqual(PersonalLiveCaptureState.Ready, adapter.CaptureState);
            session.Dispose();
        }

        [Test]
        public void Session_PrepareAutoAttempt_WhenRecognitionFailed_ResetsOnceAndReturnsReady()
        {
            var adapter = new FakeSegmentedAdapter();
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            adapter.FailActiveCaptureFromSource(new InvalidOperationException("pre-capture"));

            Assert.IsTrue(session.PrepareAutoAttempt());

            Assert.AreEqual(1, adapter.ResetCalls);
            Assert.AreEqual("GREETING", adapter.LastExpectedWord);
            Assert.AreEqual(PersonalLiveCaptureState.Ready, adapter.CaptureState);
            session.Dispose();
        }

        [Test]
        public void Session_PrepareAutoAttempt_DoesNotCreateOutcomeOrCandidate()
        {
            var adapter = new FakeSegmentedAdapter();
            var repository = new FakeRepository();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.FailActiveCaptureFromSource(new InvalidOperationException("pre-capture"));

            Assert.IsTrue(session.PrepareAutoAttempt());

            Assert.IsNull(session.LastAttempt);
            Assert.IsNull(session.LastLearningOutcome);
            Assert.AreEqual(0, repository.SaveCalls);
            Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                session.GetWordLearningStatus("GREETING"));
            Assert.AreEqual(CandidateStorageStatus.NotRequested,
                session.LastStorageResult.Status);
            session.Dispose();
        }

        [Test]
        public void Session_PrepareAutoAttempt_WhenResetFails_ReturnsFalse()
        {
            var adapter = new FakeSegmentedAdapter { ResetSucceeds = false };
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            adapter.FailActiveCaptureFromSource(new InvalidOperationException("pre-capture"));
            int expectedWordCallsBefore = adapter.SetExpectedWordCalls;

            Assert.IsFalse(session.PrepareAutoAttempt());

            Assert.AreEqual(1, adapter.ResetCalls);
            Assert.AreEqual(expectedWordCallsBefore, adapter.SetExpectedWordCalls);
            Assert.AreEqual(PersonalLiveCaptureState.Failed, adapter.CaptureState);
            Assert.IsNull(session.LastAttempt);
            session.Dispose();
        }

        [Test]
        public void Coordinator_RequestAttempt_WhenRecognitionPreparationFails_DoesNotMutateRingOrSegmenter()
        {
            var source = new FakeObservationSource();
            var adapter = new FakeSegmentedAdapter { ResetSucceeds = false };
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            adapter.FailActiveCaptureFromSource(new InvalidOperationException("pre-capture"));
            var segmenter = SavedSegmenter();
            var ring = new RawMediaPipeObservationRingBuffer(0.35d);
            ring.Add(Neutral(0d));
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                session, Service(), segmenter, ring, source, new FakeCameraContext()))
            {
                Assert.IsFalse(coordinator.RequestAttempt());

                Assert.AreEqual(1, ring.Count);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Inactive,
                    coordinator.State);
                Assert.AreEqual(string.Empty, coordinator.AttemptId);
                Assert.AreEqual(1, adapter.ResetCalls);
            }
            session.Dispose();
        }

        [Test]
        public void Coordinator_PreCaptureGenericFailure_ThenExplicitRequestAttempt_RecoversRecognitionReady()
        {
            using (var rig = new RecoveryRig(Service()))
            {
                rig.Source.EmitFailure(new InvalidOperationException("pre-capture generic"));
                Assert.AreEqual(PersonalLiveCaptureState.Failed, rig.Controller.State);
                Assert.AreEqual(PetRobotPracticeState.Ready, rig.Session.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Inactive,
                    rig.Coordinator.State);

                Assert.IsTrue(rig.Coordinator.RequestAttempt());

                Assert.AreEqual(PersonalLiveCaptureState.Ready, rig.Controller.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral,
                    rig.Coordinator.State);
                Assert.IsNull(rig.Session.LastLearningOutcome);
                Assert.AreEqual(0, rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void Coordinator_PreCaptureGenericFailure_ControllerFirst_CanReachCapturingOnNextAttempt()
        {
            using (var rig = new RecoveryRig(Service()))
            {
                Assert.IsTrue(rig.Coordinator.RequestAttempt());
                rig.Source.EmitFailure(new InvalidOperationException("controller-first pre-capture"));
                Assert.IsTrue(rig.Coordinator.RequestAttempt());

                DriveRequestedCoordinatorToCapturing(rig.Coordinator, rig.Source, 0d);

                Assert.AreEqual(PersonalLiveCaptureState.Capturing, rig.Controller.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, rig.Session.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing,
                    rig.Coordinator.State);
                Assert.AreEqual(0, rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void Coordinator_PreCaptureGenericFailure_CoordinatorFirst_CanReachCapturingOnNextAttempt()
        {
            using (var rig = new RecoveryRig(Service()))
            {
                Assert.IsTrue(rig.Coordinator.RequestAttempt());
                rig.Source.EmitFailureReverse(
                    new InvalidOperationException("coordinator-first pre-capture"));
                Assert.IsTrue(rig.Coordinator.RequestAttempt());

                DriveRequestedCoordinatorToCapturing(rig.Coordinator, rig.Source, 0d);

                Assert.AreEqual(PersonalLiveCaptureState.Capturing, rig.Controller.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, rig.Session.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing,
                    rig.Coordinator.State);
                Assert.AreEqual(0, rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void Coordinator_PreCaptureFailureDuringWaiting_RequiresNewExplicitAttemptAndThenRecovers()
        {
            using (var rig = new RecoveryRig(Service()))
            {
                Assert.IsTrue(rig.Coordinator.RequestAttempt());
                rig.Source.Emit(Neutral(0d));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral,
                    rig.Coordinator.State);
                rig.Source.EmitFailure(new InvalidOperationException("waiting failure"));

                rig.Source.Emit(Neutral(1.5d));
                rig.Source.Emit(Neutral(1.59d, handsVisible: true));
                rig.Source.Emit(Neutral(1.7d, handsVisible: true));
                rig.Source.Emit(Neutral(1.8d, handsVisible: true));
                Assert.AreEqual(PersonalLiveCaptureState.Failed, rig.Controller.State);
                Assert.AreEqual(PetRobotPracticeState.Ready, rig.Session.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed,
                    rig.Coordinator.State);
                Assert.AreEqual(0, rig.Repository.SaveCalls);

                Assert.IsTrue(rig.Coordinator.RequestAttempt());
                DriveRequestedCoordinatorToCapturing(rig.Coordinator, rig.Source, 2d);

                Assert.AreEqual(PersonalLiveCaptureState.Capturing, rig.Controller.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, rig.Session.State);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing,
                    rig.Coordinator.State);
                Assert.IsNull(rig.Session.LastLearningOutcome);
                Assert.AreEqual(0, rig.Repository.SaveCalls);
            }
        }

        [Test]
        public void Armed_StableHandEntryForPoint20Seconds_Starts()
        {
            var segmenter = ArmedSegmenter();
            RawMediaPipeNeutralAutoSegmenterDecision first = segmenter.Process(
                Neutral(1.6d, handsVisible: true), true);
            RawMediaPipeNeutralAutoSegmenterDecision beforeHold = segmenter.Process(
                Neutral(1.799d, handsVisible: true), true);
            RawMediaPipeNeutralAutoSegmenterDecision confirmed = segmenter.Process(
                Neutral(1.801d, handsVisible: true), true);

            Assert.IsFalse(first.BeginCaptureRequested);
            Assert.IsFalse(beforeHold.BeginCaptureRequested);
            Assert.IsTrue(confirmed.BeginCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
        }

        [Test]
        public void Armed_ArmsNotDownForPoint20Seconds_StartsWithoutHandLandmarker()
        {
            var segmenter = ArmedSegmenter();
            RawMediaPipeNeutralAutoSegmenterDecision first = segmenter.Process(
                Neutral(1.6d, armsDown: false), true);
            RawMediaPipeNeutralAutoSegmenterDecision beforeHold = segmenter.Process(
                Neutral(1.799d, armsDown: false), true);
            RawMediaPipeNeutralAutoSegmenterDecision confirmed = segmenter.Process(
                Neutral(1.801d, armsDown: false), true);

            Assert.IsFalse(first.BeginCaptureRequested);
            Assert.IsFalse(beforeHold.BeginCaptureRequested);
            Assert.IsTrue(confirmed.BeginCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
        }

        [Test]
        public void Armed_ForearmNotBottomCompatibleForPoint20Seconds_StartsWithoutHandLandmarker()
        {
            var segmenter = ArmedSegmenter();
            RawMediaPipeNeutralAutoSegmenterDecision first = segmenter.Process(
                Neutral(1.6d, wristMode: WristMode.Top), true);
            RawMediaPipeNeutralAutoSegmenterDecision beforeHold = segmenter.Process(
                Neutral(1.799d, wristMode: WristMode.Top), true);
            RawMediaPipeNeutralAutoSegmenterDecision confirmed = segmenter.Process(
                Neutral(1.801d, wristMode: WristMode.Top), true);

            Assert.IsFalse(first.BeginCaptureRequested);
            Assert.IsFalse(beforeHold.BeginCaptureRequested);
            Assert.IsTrue(confirmed.BeginCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
        }

        [Test]
        public void Armed_WristBaselineMismatchAlone_DoesNotStart()
        {
            UserCalibrationProfile profile = Profile(Camera(), 0.2f);
            profile.Neutral.HasWristBaseline = true;
            profile.Neutral.LeftWristNeutralBR = new Vector2(-20f, -20f);
            profile.Neutral.RightWristNeutralBR = new Vector2(20f, 20f);
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.UseSavedProfile(profile, Camera());
            Assert.IsTrue(segmenter.RequestAttempt());
            segmenter.Process(Neutral(0d), true);
            segmenter.Process(Neutral(1.5d), true);

            foreach (double timestamp in new[] { 1.6d, 1.8d, 2.0d, 2.2d })
            {
                RawMediaPipeNeutralAutoSegmenterDecision decision = segmenter.Process(
                    Neutral(timestamp, wristOffset: 0.08f), true);
                Assert.IsFalse(decision.BeginCaptureRequested);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, segmenter.State);
            }
        }

        [Test]
        public void Armed_InvalidOrAmbiguousTracking_ResetsActivationHold()
        {
            foreach (bool noPose in new[] { true, false })
            {
                var segmenter = ArmedSegmenter();
                Assert.IsFalse(segmenter.Process(
                    Neutral(1.6d, handsVisible: true), true).BeginCaptureRequested);
                Assert.IsFalse(segmenter.Process(
                    noPose ? NoPose(1.75d) : Neutral(1.75d, poseWrists: false), true)
                    .BeginCaptureRequested);
                Assert.IsFalse(segmenter.Process(
                    Neutral(1.8d, handsVisible: true), true).BeginCaptureRequested);
                Assert.IsFalse(segmenter.Process(
                    Neutral(1.999d, handsVisible: true), true).BeginCaptureRequested);
                Assert.IsTrue(segmenter.Process(
                    Neutral(2.001d, handsVisible: true), true).BeginCaptureRequested);
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
            }
        }

        [Test]
        public void Armed_InterruptedActivationRequiresFreshPoint20Seconds()
        {
            var segmenter = ArmedSegmenter();
            Assert.IsFalse(segmenter.Process(
                Neutral(1.6d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(
                Neutral(1.75d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(Neutral(1.76d), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(
                Neutral(1.8d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsFalse(segmenter.Process(
                Neutral(1.999d, handsVisible: true), true).BeginCaptureRequested);
            Assert.IsTrue(segmenter.Process(
                Neutral(2.001d, handsVisible: true), true).BeginCaptureRequested);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, segmenter.State);
        }

        static DateTime FixedClock() => new DateTime(2026, 8, 16, 0, 0, 0, DateTimeKind.Utc);

        PetRobotNeutralCalibrationProfileService Service() =>
            new PetRobotNeutralCalibrationProfileService(LearnerId, tempRoot, null, FixedClock);

        PetRobotNeutralAutoCaptureCoordinator CreateSavedCoordinator(
            out FakeObservationSource source,
            out FakeSegmentedAdapter adapter,
            out PetRobotSignPracticeSession session,
            out RawMediaPipeNeutralAutoSegmenter segmenter)
        {
            PetRobotNeutralCalibrationProfileService service = Service();
            source = new FakeObservationSource();
            adapter = new FakeSegmentedAdapter();
            session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            segmenter.UseSavedProfile(Profile(Camera(), 0.2f), Camera());
            var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                session, service, segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext());
            SetPrivateField(coordinator, "profileResolvedForCamera", true);
            return coordinator;
        }

        // ---- F53: the upright convention on the calibration/segmentation entry path ------
        //
        // These exercise the ring buffer, the segmenter and the coordinator with the F51
        // upright convention. Each of the first three fails against the two old
        // LiveMediaPipe-only guards, which is exactly the reported start blocker.

        static Step9DInferenceFrameGeometry UprightGeometry(
            bool upright, int? rotation, bool? mirrored, int width)
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                width, 480, 30, width, 480, rotation, mirrored,
                width, 480, width, 480, "RGBA32",
                Step9DPreInferenceTransformState.None,
                upright ? Step9DPreInferenceTransformState.Applied
                        : Step9DPreInferenceTransformState.None,
                0, Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, 0L, 0, upright ? "top-down" : "bottom-up");
        }

        static Step9DMergedInferenceGeometry UprightMerged(
            bool handUpright = true, bool poseUpright = true,
            int? rotation = 0, bool? mirrored = false, int poseWidth = 640)
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(
                    UprightGeometry(handUpright, rotation, mirrored, 640)),
                Step9DGeometryAcquisitionResult.ForAvailable(
                    UprightGeometry(poseUpright, rotation, mirrored, poseWidth)));
        }

        static RawMediaPipeObservation WithConvention(
            RawMediaPipeObservation source, RawObservationConvention convention,
            Step9DMergedInferenceGeometry geometry)
        {
            return new RawMediaPipeObservation(
                source.TimestampSeconds, convention,
                source.Pose, source.LeftHand, source.RightHand, geometry);
        }

        /// <summary>The same synthetic body, re-tagged as a fully valid upright observation.</summary>
        static RawMediaPipeObservation AsUpright(RawMediaPipeObservation source)
        {
            return WithConvention(source,
                RawObservationConvention.LiveMediaPipeUprightTexture, UprightMerged());
        }

        [Test]
        public void F53_Ring_AcceptsUprightAndStillRefusesUnbackedOrMixedObservations()
        {
            var ring = new RawMediaPipeObservationRingBuffer(5d);
            ring.Add(AsUpright(Neutral(0d)));
            ring.Add(AsUpright(Neutral(0.1d)));
            Assert.AreEqual(2, ring.Count, "upright observations must reach the pre-roll");

            var upright = RawObservationConvention.LiveMediaPipeUprightTexture;
            Assert.Throws<ArgumentException>(() => new RawMediaPipeObservationRingBuffer(5d)
                .Add(WithConvention(Neutral(0d), upright, null)), "missing geometry");
            Assert.Throws<ArgumentException>(() => new RawMediaPipeObservationRingBuffer(5d)
                .Add(WithConvention(Neutral(0d), upright, UprightMerged(poseUpright: false))),
                "mixed provider metadata");
            Assert.Throws<ArgumentException>(() => new RawMediaPipeObservationRingBuffer(5d)
                .Add(WithConvention(Neutral(0d), upright,
                     UprightMerged(rotation: null, mirrored: null))), "unknown camera geometry");
            Assert.Throws<ArgumentException>(() => new RawMediaPipeObservationRingBuffer(5d)
                .Add(WithConvention(Neutral(0d), upright, UprightMerged(rotation: 90))),
                "unsupported rotation");
            Assert.Throws<ArgumentException>(() => new RawMediaPipeObservationRingBuffer(5d)
                .Add(WithConvention(Neutral(0d), upright, UprightMerged(poseWidth: 320))),
                "contract-incompatible provider geometries");

            var mixing = new RawMediaPipeObservationRingBuffer(5d);
            mixing.Add(Neutral(0d));
            Assert.Throws<ArgumentException>(() => mixing.Add(AsUpright(Neutral(0.1d))),
                "the convention must stay stable inside one ring");
            Assert.Throws<ArgumentException>(() => ring.Add(AsUpright(Neutral(0.1d))),
                "ring timestamps must still be strictly increasing");
        }

        [Test]
        public void F53_Segmenter_UprightObservationsCalibrateArmCaptureAndFinalize()
        {
            var fresh = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            fresh.RequireCalibration(Camera(), "missing");
            Assert.IsTrue(fresh.RequestAttempt());
            Assert.IsNull(fresh.Process(AsUpright(Neutral(0d)), true).CalibrationProfile);
            UserCalibrationProfile calibrated =
                fresh.Process(AsUpright(Neutral(2.5d)), true).CalibrationProfile;
            Assert.IsNotNull(calibrated, "upright neutral frames must calibrate");
            Assert.AreEqual(LearnerId, calibrated.UserProfileId);

            var saved = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            saved.UseSavedProfile(Profile(Camera(), 0.2f), Camera());
            Assert.IsTrue(saved.RequestAttempt());
            saved.Process(AsUpright(Neutral(0d)), true);
            saved.Process(AsUpright(Neutral(1.5d)), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, saved.State);
            saved.Process(AsUpright(Neutral(1.59d, handsVisible: true)), true);
            saved.Process(AsUpright(Neutral(1.7d, handsVisible: true)), true);
            saved.Process(AsUpright(Neutral(1.8d, handsVisible: true)), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, saved.State);

            bool finalized = false;
            for (int frame = 1; frame <= 60 && !finalized; frame++)
                finalized = saved.Process(
                    AsUpright(Neutral(1.8d + frame * FrameStep)), true).FinalizeCaptureRequested;
            Assert.IsTrue(finalized, "the end hold must finalize an upright capture");

            Assert.Throws<ArgumentException>(() => new RawMediaPipeNeutralAutoSegmenter(LearnerId)
                .Process(WithConvention(Neutral(0d),
                    RawObservationConvention.LiveMediaPipeUprightTexture, null), true),
                "upright without geometry is still refused");
        }

        [Test]
        public void F53_Coordinator_UprightAttempt_CalibratesStartsAndFinalizesOnce()
        {
            PetRobotNeutralCalibrationProfileService service = Service();
            var source = new FakeObservationSource();
            var segmenter = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            var adapter = new FakeSegmentedAdapter();
            var session = new PetRobotSignPracticeSession(adapter, new FakeRepository());
            using (var coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                session, service, segmenter, new RawMediaPipeObservationRingBuffer(0.35d),
                source, new FakeCameraContext()))
            {
                Assert.IsTrue(coordinator.RequestAttempt());
                source.Emit(AsUpright(Neutral(0d)));
                source.Emit(AsUpright(Neutral(2.5d)));
                Assert.IsTrue(File.Exists(service.ProfilePath),
                    "a fresh neutral profile must be calibrated and saved from upright frames");
                Assert.IsTrue(coordinator.IsProfileLoaded);

                source.Emit(AsUpright(Neutral(3d)));
                source.Emit(AsUpright(Neutral(4.5d)));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Armed, coordinator.State);
                source.Emit(AsUpright(Neutral(4.59d, handsVisible: true)));
                source.Emit(AsUpright(Neutral(4.7d, handsVisible: true)));
                source.Emit(AsUpright(Neutral(4.8d, handsVisible: true)));
                Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
                Assert.AreEqual(PetRobotPracticeState.Capturing, session.State,
                    "the session must actually have started one auto expression");

                for (int frame = 1; frame <= 60 &&
                     (coordinator.State == RawMediaPipeNeutralAutoSegmenterState.Capturing ||
                      coordinator.State == RawMediaPipeNeutralAutoSegmenterState.EndCandidate);
                     frame++)
                    source.Emit(AsUpright(Neutral(4.8d + frame * FrameStep)));

                Assert.AreNotEqual(PetRobotPracticeState.Capturing, session.State,
                    "the attempt must finalize");
                Assert.AreNotEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, coordinator.State,
                    "a second finalize or a rejected observation would have failed the segmenter");
                Assert.IsTrue(string.IsNullOrEmpty(coordinator.LastFailure), coordinator.LastFailure);
            }
            session.Dispose();
        }

        [Test]
        public void F53_PolicyIsTheOneRuleTheCaptureControllerAlsoApplies()
        {
            MethodInfo controllerPredicate = typeof(PersonalLiveRecognitionController).GetMethod(
                "IsAcceptedCaptureConvention", BindingFlags.Static | BindingFlags.NonPublic);
            Assert.IsNotNull(controllerPredicate, "the controller predicate must still exist");

            var upright = RawObservationConvention.LiveMediaPipeUprightTexture;
            var foreign = new RawObservationConvention(
                "some_other_normalized_image_v9",
                RawObservationCoordinateSpace.MediaPipeNormalizedImage,
                "x-right/y-up", "direct", "not mirrored");
            var cases = new[]
            {
                new { Observation = Neutral(0d), Expected = true },
                new { Observation = AsUpright(Neutral(0d)), Expected = true },
                new { Observation = WithConvention(Neutral(0d), upright, null), Expected = false },
                new { Observation = WithConvention(Neutral(0d), upright,
                          UprightMerged(handUpright: false)), Expected = false },
                new { Observation = WithConvention(Neutral(0d), upright,
                          UprightMerged(mirrored: true)), Expected = false },
                new { Observation = WithConvention(Neutral(0d), upright,
                          UprightMerged(rotation: null, mirrored: null)), Expected = false },
                new { Observation = WithConvention(Neutral(0d), foreign, UprightMerged()),
                      Expected = false },
            };
            foreach (var item in cases)
            {
                Assert.AreEqual(item.Expected,
                    LiveObservationConventionPolicy.IsAccepted(item.Observation),
                    item.Observation.Convention.Id);
                Assert.AreEqual(item.Expected,
                    (bool)controllerPredicate.Invoke(null, new object[] { item.Observation }),
                    "the controller must apply the identical rule");
            }
            Assert.IsFalse(LiveObservationConventionPolicy.IsAccepted(null));
        }

        static void DriveCoordinatorToCapturing(
            PetRobotNeutralAutoCaptureCoordinator coordinator,
            FakeObservationSource source,
            double start)
        {
            Assert.IsTrue(coordinator.RequestAttempt());
            source.Emit(Neutral(start));
            source.Emit(Neutral(start + 1.5d));
            source.Emit(Neutral(start + 1.59d, handsVisible: true));
            source.Emit(Neutral(start + 1.7d, handsVisible: true));
            source.Emit(Neutral(start + 1.8d, handsVisible: true));
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
        }

        static void DriveRequestedCoordinatorToCapturing(
            PetRobotNeutralAutoCaptureCoordinator coordinator,
            FakeObservationSource source,
            double start)
        {
            source.Emit(Neutral(start));
            source.Emit(Neutral(start + 1.5d));
            source.Emit(Neutral(start + 1.59d, handsVisible: true));
            source.Emit(Neutral(start + 1.7d, handsVisible: true));
            source.Emit(Neutral(start + 1.8d, handsVisible: true));
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, coordinator.State);
        }

        static void AssertRecoveredTechnicalFailure(RecoveryRig rig, string expectedFailureText)
        {
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Failed, rig.Coordinator.State);
            Assert.AreEqual(PetRobotPracticeState.Feedback, rig.Session.State);
            Assert.AreEqual(PersonalLiveCaptureState.Failed, rig.Controller.State);
            Assert.IsNotNull(rig.Controller.Failure);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, rig.Controller.Failure.Code);
            StringAssert.Contains(expectedFailureText, rig.Controller.Failure.Message);
            Assert.IsNotNull(rig.Controller.PrimaryFailure);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing,
                rig.Controller.PrimaryFailure.StateBeforeFailure);
            Assert.IsNotNull(rig.Session.LastAttempt);
            Assert.AreEqual(rig.Controller.CaptureSessionId,
                rig.Session.LastAttempt.CaptureSessionId);
            Assert.AreEqual(rig.Controller.ExpectedWord, rig.Session.LastAttempt.ExpectedWord);
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                rig.Session.LastAttempt.Outcome.Kind);
            Assert.AreSame(rig.Controller.Failure, rig.Session.LastAttempt.Failure);
            Assert.AreSame(rig.Controller.PrimaryFailure, rig.Session.LastAttempt.PrimaryFailure);
            Assert.AreEqual(CandidateStorageStatus.NotRequested,
                rig.Session.LastStorageResult.Status);
            Assert.AreEqual(0, rig.Repository.SaveCalls);
            Assert.AreEqual(2, rig.SessionChangedCount);
        }

        static void SetPrivateField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, "missing test-only field " + fieldName);
            field.SetValue(target, value);
        }

        void AssertNoTransactionResidue()
        {
            if (!Directory.Exists(tempRoot)) return;
            Assert.IsEmpty(Directory.GetFiles(tempRoot, "*.tmp", SearchOption.AllDirectories));
            Assert.IsEmpty(Directory.GetFiles(tempRoot, "*.bak", SearchOption.AllDirectories));
        }

        static RawMediaPipeNeutralAutoSegmenter SavedSegmenter()
        {
            var value = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            value.UseSavedProfile(Profile(Camera(), 0.2f), Camera());
            return value;
        }

        static RawMediaPipeNeutralAutoSegmenter SavedHandsVisibleSegmenter()
        {
            UserCalibrationProfile profile = Profile(Camera(), 0.2f);
            profile.Neutral.HandsVisible = true;
            profile.Neutral.BottomExitAllowed = false;
            profile.Neutral.HasWristBaseline = true;
            profile.Neutral.LeftWristNeutralBR = new Vector2(-0.5f, 2f);
            profile.Neutral.RightWristNeutralBR = new Vector2(0.5f, 2f);
            var value = new RawMediaPipeNeutralAutoSegmenter(LearnerId);
            value.UseSavedProfile(profile, Camera());
            return value;
        }

        static RawMediaPipeNeutralAutoSegmenter ArmedSegmenter()
        {
            RawMediaPipeNeutralAutoSegmenter value = SavedSegmenter();
            value.RequestAttempt();
            value.Process(Neutral(0d), true);
            value.Process(Neutral(1.5d), true);
            return value;
        }

        static RawMediaPipeNeutralAutoSegmenter CapturingSegmenter()
        {
            RawMediaPipeNeutralAutoSegmenter value = ArmedSegmenter();
            value.Process(Neutral(1.59d, handsVisible: true), true);
            value.Process(Neutral(1.7d, handsVisible: true), true);
            value.Process(Neutral(1.8d, handsVisible: true), true);
            Assert.AreEqual(RawMediaPipeNeutralAutoSegmenterState.Capturing, value.State);
            return value;
        }

        /// <summary>
        /// One frame of the ~23 Hz cadence the production camera actually delivers. The
        /// endpoint continuity bound is 0.2 s, so this is continuous evidence.
        /// </summary>
        const double FrameStep = 0.043d;

        /// <summary>
        /// Feeds <paramref name="frames"/> observed neutral frames and asserts that none of
        /// them finalized, i.e. the capture is not truncated before the intended margin.
        /// </summary>
        static void HoldNeutralWithoutFinalize(
            RawMediaPipeNeutralAutoSegmenter segmenter,
            double start,
            int frames,
            Func<int, float> wristOffset = null)
        {
            for (int index = 1; index <= frames; index++)
                Assert.IsFalse(segmenter.Process(
                    Neutral(start + index * FrameStep,
                            wristOffset: wristOffset == null ? 0f : wristOffset(index)), true)
                    .FinalizeCaptureRequested,
                    "finalized after only " + (index * FrameStep) + " s of neutral");
        }

        enum WristMode { Bottom, Side, Top }

        static RawMediaPipeObservation Neutral(
            double timestamp,
            bool armsDown = true,
            bool poseWrists = true,
            bool handsVisible = false,
            float wristOffset = 0f,
            float shoulderWidth = 0.2f,
            WristMode wristMode = WristMode.Bottom,
            bool? leftPoseWrist = null,
            bool? rightPoseWrist = null)
        {
            double center = 0.5d;
            double leftShoulderX = center - shoulderWidth * 0.5d;
            double rightShoulderX = center + shoulderWidth * 0.5d;
            var coordinates = NaNs(RawMediaPipeObservation.PoseLandmarkCount);
            var valid = new bool[RawMediaPipeObservation.PoseLandmarkCount];
            Write(coordinates, valid, 11, leftShoulderX, 0.3d, 0d);
            Write(coordinates, valid, 12, rightShoulderX, 0.3d, 0d);
            Write(coordinates, valid, 13, armsDown ? leftShoulderX : leftShoulderX - 0.2d,
                armsDown ? 0.1d : 0.3d, 0d);
            Write(coordinates, valid, 14, armsDown ? rightShoulderX : rightShoulderX + 0.2d,
                armsDown ? 0.1d : 0.3d, 0d);
            bool writeLeftWrist = leftPoseWrist ?? poseWrists;
            bool writeRightWrist = rightPoseWrist ?? poseWrists;
            if (writeLeftWrist || writeRightWrist)
            {
                double y = wristMode == WristMode.Top ? 0.4d : -0.1d;
                double side = wristMode == WristMode.Side ? 0.25d : 0d;
                if (writeLeftWrist)
                    Write(coordinates, valid, 15, leftShoulderX + wristOffset - side, y, 0d);
                if (writeRightWrist)
                    Write(coordinates, valid, 16, rightShoulderX + wristOffset + side, y, 0d);
            }
            RawLandmarkObservationSet hand = handsVisible ? VisibleHand() : RawLandmarkObservationSet.Absent(21);
            return new RawMediaPipeObservation(
                timestamp, RawObservationConvention.LiveMediaPipe,
                new RawLandmarkObservationSet(33, coordinates, valid, new bool[33], new bool[33]),
                hand, handsVisible ? VisibleHand() : RawLandmarkObservationSet.Absent(21));
        }

        static RawMediaPipeObservation RecognitionObservation(double timestamp)
        {
            var poseCoordinates = NaNs(33);
            var poseValid = new bool[33];
            Write(poseCoordinates, poseValid, 11, -1d, 0d, 0d);
            Write(poseCoordinates, poseValid, 12, 1d, 0d, 0d);
            Write(poseCoordinates, poseValid, 23, -1d, 2d, 0d);
            Write(poseCoordinates, poseValid, 24, 1d, 2d, 0d);
            Write(poseCoordinates, poseValid, 13, -1.2d, -0.5d, 0.1d);
            Write(poseCoordinates, poseValid, 14, 1.2d, -0.5d, 0.1d);
            Write(poseCoordinates, poseValid, 15, -1.4d, -1d, 0.2d);
            Write(poseCoordinates, poseValid, 16, 1.4d, -1d, 0.2d);
            return new RawMediaPipeObservation(
                timestamp, RawObservationConvention.LiveMediaPipe,
                new RawLandmarkObservationSet(33, poseCoordinates, poseValid, new bool[33], new bool[33]),
                VisibleHand(), RawLandmarkObservationSet.Absent(21));
        }

        static RawMediaPipeObservation NoPose(double timestamp) =>
            new RawMediaPipeObservation(
                timestamp, RawObservationConvention.LiveMediaPipe,
                RawLandmarkObservationSet.Absent(RawMediaPipeObservation.PoseLandmarkCount),
                RawLandmarkObservationSet.Absent(RawMediaPipeObservation.HandLandmarkCount),
                RawLandmarkObservationSet.Absent(RawMediaPipeObservation.HandLandmarkCount));

        static RawLandmarkObservationSet VisibleHand()
        {
            var coordinates = NaNs(21);
            var valid = new bool[21];
            foreach (int landmark in new[] { 0, 4, 8, 12, 16, 20 })
                Write(coordinates, valid, landmark, 0.4d + landmark * 0.005d, 0.6d, 0d);
            return new RawLandmarkObservationSet(21, coordinates, valid, new bool[21], new bool[21]);
        }

        static double[] NaNs(int count)
        {
            var result = new double[count * 3];
            for (int index = 0; index < result.Length; index++) result[index] = double.NaN;
            return result;
        }

        static void Write(double[] coordinates, bool[] valid, int landmark, double x, double y, double z)
        {
            int offset = landmark * 3;
            coordinates[offset] = x;
            coordinates[offset + 1] = y;
            coordinates[offset + 2] = z;
            valid[landmark] = true;
        }

        static CameraFingerprint Camera() => new CameraFingerprint
        {
            DeviceName = "test-camera",
            Width = 640,
            Height = 480,
            Fps = 30f,
            RotationDegrees = 0,
            PreviewMirrored = true,
        };

        static UserCalibrationProfile Profile(CameraFingerprint camera, float shoulderWidth) =>
            new UserCalibrationProfile
            {
                SchemaVersion = 1,
                UserProfileId = LearnerId,
                Camera = camera,
                BodyScale = new BodyScaleData { ShoulderWidthNorm = shoulderWidth },
                Neutral = new NeutralPoseData
                {
                    HandsVisible = false,
                    BottomExitAllowed = true,
                    HasArmVerticalBaseline = true,
                    LeftArmVerticalNeutral = 0f,
                    RightArmVerticalNeutral = 0f,
                    HasWristBaseline = false,
                    MotionNoiseFloor = 0.002f,
                },
                Quality = new CalibrationQuality { Status = "good", ValidFrameRatio = 1f },
            };

        static Step9CAmbiguousHandIdentityException Ambiguity(int providerFrameIndex)
        {
            double timestamp = providerFrameIndex / 30d;
            var diagnostic = new Step9CHandIdentityDiagnostic(
                providerFrameIndex: providerFrameIndex,
                providerTimestampSeconds: timestamp,
                handFrameTimestampSeconds: timestamp,
                sourceHandCount: 2,
                sourceCategory0: "Left",
                sourceScore0: 0.91d,
                sourceCategory1: "Left",
                sourceScore1: 0.99d,
                hand0WristX: 0.25d,
                hand0WristY: 0.5d,
                hand1WristX: 0.75d,
                hand1WristY: 0.5d,
                poseLeftWristX: double.NaN,
                poseLeftWristY: double.NaN,
                poseRightWristX: double.NaN,
                poseRightWristY: double.NaN,
                poseLeftWristVisibility: double.NaN,
                poseRightWristVisibility: double.NaN,
                directAssignmentCost: double.NaN,
                swappedAssignmentCost: double.NaN,
                bestAssignmentMaxDistance: double.NaN,
                assignmentMargin: double.NaN,
                poseWristSeparation: double.NaN,
                selectedHand0Identity: string.Empty,
                outcome: Step9CHandIdentityOutcome.Ambiguous,
                reason: "same_frame_pose_wrists_unavailable_or_low_visibility");
            return new Step9CAmbiguousHandIdentityException(diagnostic);
        }

        static PersonalLiveRecognitionController ReadyController(out FakeObservationSource source)
        {
            source = new FakeObservationSource();
            var controller = new PersonalLiveRecognitionController(
                source, new FakeRuntimeFactory(new FakeRuntime(SuccessRouting())));
            Assert.IsTrue(controller.Initialize(new PersonalLiveRuntimePaths("global", "threshold", "profile")));
            return controller;
        }

        static PersonalLiveRoutingResult SuccessRouting()
        {
            var scores = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["GREETING"] = 0.1d, ["LOVE"] = 0.3d, ["THANK_YOU"] = 0.4d,
            };
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal)
            {
                ["GREETING"] = true, ["LOVE"] = false, ["THANK_YOU"] = false,
            };
            return new PersonalLiveRoutingResult(
                "GREETING", true, true, true, scores, accepts, 0.1d, "prototype-GREETING", Array.Empty<string>());
        }

        static PracticeRecognitionAttempt Attempt(
            GuidedQuestOutcomeKind kind,
            string captureSessionId = null,
            string expectedWord = "GREETING")
        {
            string predicted = kind == GuidedQuestOutcomeKind.WrongWord ? "LOVE" : "GREETING";
            bool completed = kind != GuidedQuestOutcomeKind.TechnicalFailure;
            bool strict = kind == GuidedQuestOutcomeKind.StrictSuccess;
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                completed ? PersonalLiveCaptureState.Completed : PersonalLiveCaptureState.Failed,
                completed,
                expectedWord,
                completed ? predicted : string.Empty,
                strict,
                strict,
                strict,
                !completed,
                !completed);
            return new PracticeRecognitionAttempt(
                captureSessionId ?? Guid.NewGuid().ToString("D"), expectedWord, predicted,
                completed ? PersonalLiveCaptureState.Completed : PersonalLiveCaptureState.Failed,
                outcome, null, null, "candidate", "personal", 0.5d, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = completed ? "prototype" : string.Empty });
        }

        static string ReadProject(string relativePath) => File.ReadAllText(Path.Combine(
            Directory.GetParent(Application.dataPath).FullName,
            relativePath.Replace('/', Path.DirectorySeparatorChar)));

        sealed class FakeObservationSource : IPersonalLiveObservationSource
        {
            Action<RawMediaPipeObservation> observations;
            Action<Exception> failures;
            public bool IsBound { get; set; } = true;
            public int SubscriberCount =>
                (observations == null ? 0 : observations.GetInvocationList().Length) +
                (failures == null ? 0 : failures.GetInvocationList().Length);
            public event Action<RawMediaPipeObservation> ObservationArrived
            {
                add { observations += value; }
                remove { observations -= value; }
            }
            public event Action<Exception> SourceFailed
            {
                add { failures += value; }
                remove { failures -= value; }
            }
            public void Emit(RawMediaPipeObservation value) => observations?.Invoke(value);
            public void EmitFailure(Exception exception) => failures?.Invoke(exception);
            public void EmitFailureReverse(Exception exception)
            {
                if (failures == null) return;
                Delegate[] invocation = failures.GetInvocationList();
                for (int index = invocation.Length - 1; index >= 0; index--)
                    ((Action<Exception>)invocation[index])(exception);
            }
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime { }

        sealed class FakeRuntime : IPersonalLiveRecognitionRuntime
        {
            readonly PersonalLiveRoutingResult result;
            readonly string activeProfileId;
            public bool IsPersonalActive => true;
            public string ActiveProfileId => activeProfileId;
            public FakeRuntime(
                PersonalLiveRoutingResult result,
                string activeProfileId = "test-personal-profile")
            {
                this.result = result;
                this.activeProfileId = activeProfileId;
            }
            public PersonalLiveRoutingResult Recognize(
                string sessionId, RecognitionSequence sequence, string expectedWord) => result;
        }

        sealed class FakeRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            readonly FakeRuntime runtime;
            public FakeRuntimeFactory(FakeRuntime runtime) { this.runtime = runtime; }
            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) => new FakeGlobalRuntime();
            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths) => runtime;
        }

        sealed class FakeSegmentedAdapter : ISegmentedPersonalLivePracticeRecognitionAdapter
        {
            public PracticeRecognitionAttempt Next;
            public PracticeRecognitionAttempt FailureNext;
            public int ManualBeginCalls;
            public int ManualFinishCalls;
            public int AutoBeginCalls;
            public int AutoFinishCalls;
            public int FailureRecoveryCalls;
            public int ResetCalls;
            public int SetExpectedWordCalls;
            public bool ResetSucceeds = true;
            public string LastExpectedWord { get; private set; } = string.Empty;
            public PersonalLiveCaptureState CaptureState { get; private set; } = PersonalLiveCaptureState.Ready;
            public string CaptureSessionId => Next != null
                ? Next.CaptureSessionId
                : FailureNext == null ? string.Empty : FailureNext.CaptureSessionId;
            public string ActiveProfileId => "personal";
            public bool CandidateRouteActive => true;
            public bool SetExpectedWord(string word)
            {
                SetExpectedWordCalls++;
                LastExpectedWord = word;
                return true;
            }
            public bool BeginCapture()
            {
                ManualBeginCalls++;
                CaptureState = PersonalLiveCaptureState.Capturing;
                return true;
            }
            public PracticeRecognitionAttempt FinalizeCapture()
            {
                ManualFinishCalls++;
                CaptureState = PersonalLiveCaptureState.Completed;
                return Next;
            }
            public bool BeginCaptureWithPreRoll(IReadOnlyList<RawMediaPipeObservation> preRoll)
            {
                AutoBeginCalls++;
                CaptureState = PersonalLiveCaptureState.Capturing;
                return preRoll != null && preRoll.Count > 0;
            }
            public PracticeRecognitionAttempt FinalizeCaptureWithTrailingTrim(double trailingTrimSeconds)
            {
                AutoFinishCalls++;
                CaptureState = PersonalLiveCaptureState.Completed;
                return Next;
            }
            public PracticeRecognitionAttempt FailActiveCaptureFromSource(Exception exception)
            {
                FailureRecoveryCalls++;
                CaptureState = PersonalLiveCaptureState.Failed;
                return FailureNext ?? Attempt(GuidedQuestOutcomeKind.TechnicalFailure);
            }
            public bool ResetCapture()
            {
                ResetCalls++;
                if (!ResetSucceeds) return false;
                CaptureState = PersonalLiveCaptureState.Ready;
                return true;
            }
        }

        sealed class RecoveryRig : IDisposable
        {
            readonly GameObject host;
            public FakeObservationSource Source { get; }
            public PersonalLiveRecognitionController Controller { get; }
            public PersonalLivePracticeRecognitionAdapter Adapter { get; }
            public FakeRepository Repository { get; }
            public PetRobotSignPracticeSession Session { get; }
            public RawMediaPipeNeutralAutoSegmenter Segmenter { get; }
            public PetRobotNeutralAutoCaptureCoordinator Coordinator { get; }
            public int SessionChangedCount { get; private set; }

            public RecoveryRig(PetRobotNeutralCalibrationProfileService service)
            {
                Source = new FakeObservationSource();
                Controller = new PersonalLiveRecognitionController(
                    Source,
                    new FakeRuntimeFactory(new FakeRuntime(
                        SuccessRouting(), PersonalMultiPrototypeRuntimeProfileLoader.ProfileId)),
                    enableCandidateTransientAmbiguityTolerance: true);
                Assert.IsTrue(Controller.Initialize(
                    new PersonalLiveRuntimePaths("global", "threshold", "profile")));

                host = new GameObject("Step10J_v1_2_RecoveryRig");
                host.SetActive(false);
                PersonalLiveRecognitionHarness harness =
                    host.AddComponent<PersonalLiveRecognitionHarness>();
                SetPrivateField(harness, "profileMode",
                    PersonalLiveProfileMode.MultiPrototypeCandidate);
                SetPrivateField(harness, "controller", Controller);
                Adapter = new PersonalLivePracticeRecognitionAdapter(harness);
                Repository = new FakeRepository();
                Session = new PetRobotSignPracticeSession(Adapter, Repository);
                Session.Changed += OnSessionChanged;
                Segmenter = SavedSegmenter();
                Coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                    Session,
                    service,
                    Segmenter,
                    new RawMediaPipeObservationRingBuffer(0.35d),
                    Source,
                    new FakeCameraContext());
                SetPrivateField(Coordinator, "profileResolvedForCamera", true);
            }

            public void Dispose()
            {
                Coordinator.Dispose();
                Session.Changed -= OnSessionChanged;
                Session.Dispose();
                Controller.Dispose();
                UnityEngine.Object.DestroyImmediate(host);
            }

            void OnSessionChanged() => SessionChangedCount++;
        }

        sealed class FakeRepository : IPersonalizationCandidateRepository
        {
            public int SaveCalls;
            public PersonalizationCandidateEligibility LastEligibility;
            public string RootDirectory => tempRootStatic;
            public string LearnerId => PetRobotNeutralAutoSegmentationV1Tests.LearnerId;
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;
            static string tempRootStatic => string.Empty;
            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                LastEligibility = request.Eligibility;
                return new CandidateSaveResult(CandidateStorageStatus.Saved, "candidate", "saved", Counts);
            }
        }

        sealed class FakeCameraContext : IPetRobotNeutralCameraContext
        {
            public bool IsReady => true;
            public bool TrySnapshotCameraFingerprint(out CameraFingerprint fingerprint)
            {
                fingerprint = Camera();
                return true;
            }
        }

        public enum FaultPoint { TempWrite, TempValidation, Move, Replace, PostValidation }

        sealed class FaultFileSystem : INeutralCalibrationProfileFileSystem
        {
            readonly SystemNeutralCalibrationProfileFileSystem inner =
                new SystemNeutralCalibrationProfileFileSystem();
            readonly FaultPoint point;
            readonly string finalPath;
            bool replaced;

            public FaultFileSystem(FaultPoint point, string finalPath)
            {
                this.point = point;
                this.finalPath = finalPath;
            }

            public bool FileExists(string path) => inner.FileExists(path);
            public void CreateDirectory(string path) => inner.CreateDirectory(path);
            public string ReadAllText(string path)
            {
                if (point == FaultPoint.TempValidation && path.EndsWith(".tmp", StringComparison.Ordinal))
                    return "{}";
                if (point == FaultPoint.PostValidation && replaced && path == finalPath)
                    return "{}";
                return inner.ReadAllText(path);
            }
            public void WriteAllText(string path, string contents)
            {
                if (point == FaultPoint.TempWrite && path.EndsWith(".tmp", StringComparison.Ordinal))
                    throw new IOException("injected temp write failure");
                inner.WriteAllText(path, contents);
            }
            public void MoveFile(string source, string destination)
            {
                if (point == FaultPoint.Move) throw new IOException("injected move failure");
                inner.MoveFile(source, destination);
            }
            public void ReplaceFile(string source, string destination, string backup)
            {
                if (point == FaultPoint.Replace) throw new IOException("injected replace failure");
                inner.ReplaceFile(source, destination, backup);
                replaced = true;
            }
            public void DeleteFile(string path) => inner.DeleteFile(path);
        }
    }
}
