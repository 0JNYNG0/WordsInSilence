using System;
using NUnit.Framework;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0e: a candidate opt-in capture must survive EVERY typed
    /// Step9CAmbiguousHandIdentityException, not only the first one.
    ///
    /// The v1.0d3 live run failed exactly here. The THANK_YOU take reached
    /// ResetCapture, SetExpectedWord and BeginCapture, then a second typed ambiguity
    /// exhausted the one-event tolerance, was promoted to a fatal BridgeFailure, and
    /// cleared the observations already buffered for that capture. The take ended in
    /// CaptureState=Failed and no common-2D record was ever produced.
    ///
    /// A typed ambiguity is a frame the resolver could not attribute to a hand. The only
    /// safe response is to drop it: the capture keeps running, nothing is synthesized in
    /// its place and no handedness is guessed. Everything else is deliberately unchanged
    /// - opt-in false is still fail-fast, generic and geometry failures are still fatal,
    /// and a capture that ends with no valid observation still fails EmptyCapture.
    ///
    /// These tests drive the real <see cref="PersonalLiveRecognitionController"/>
    /// through a plain event-pump source. Only the injected collaborators are doubles;
    /// the branch under test is production code.
    /// </summary>
    public sealed class PersonalLiveRecognitionAmbiguityRecoveryV1eTests
    {
        sealed class FakeObservationSource : IPersonalLiveObservationSource
        {
            Action<RawMediaPipeObservation> observations;
            Action<Exception> failures;

            public bool IsBound { get; set; } = true;

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

            public int SubscriberCount =>
                (observations == null ? 0 : observations.GetInvocationList().Length) +
                (failures == null ? 0 : failures.GetInvocationList().Length);

            public void Emit(RawMediaPipeObservation observation) =>
                observations?.Invoke(observation);
            public void Fail(Exception exception) => failures?.Invoke(exception);
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime
        {
        }

        /// <summary>
        /// Records whether recognition was reached. It throws if it ever is, so a test
        /// asserting that a capture never reaches the model cannot pass by accident.
        /// </summary>
        sealed class FakeRuntime : IPersonalLiveRecognitionRuntime
        {
            public bool IsPersonalActive => true;
            public string ActiveProfileId => "v1-0e-ambiguity-recovery-fixture";
            public int RecognizeCallCount { get; private set; }

            public PersonalLiveRoutingResult Recognize(
                string sessionId, RecognitionSequence sequence, string expectedWord)
            {
                RecognizeCallCount++;
                throw new InvalidOperationException(
                    "this fixture never authorizes model recognition");
            }
        }

        sealed class FakeRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            public FakeRuntime Runtime { get; } = new FakeRuntime();

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) =>
                new FakeGlobalRuntime();

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths) =>
                Runtime;
        }

        static FakeRuntimeFactory lastFactory;

        static PersonalLiveRecognitionController ReadyController(
            FakeObservationSource source, bool candidateTolerance)
        {
            lastFactory = new FakeRuntimeFactory();
            var controller = new PersonalLiveRecognitionController(
                source,
                lastFactory,
                enableCandidateTransientAmbiguityTolerance: candidateTolerance);
            Assert.IsTrue(controller.Initialize(new PersonalLiveRuntimePaths(
                "global-templates", "global-threshold", "personal-profile")));
            return controller;
        }

        static RawMediaPipeObservation Observation(double timestamp) =>
            new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.LiveMediaPipe,
                RawLandmarkObservationSet.Absent(33),
                RawLandmarkObservationSet.Absent(21),
                RawLandmarkObservationSet.Absent(21));

        static RawMediaPipeObservation ForeignConventionObservation(double timestamp) =>
            new RawMediaPipeObservation(
                timestamp,
                new RawObservationConvention(
                    "v1_0e_fixture_foreign_convention",
                    RawObservationCoordinateSpace.MediaPipeNormalizedImage,
                    "not the approved live convention",
                    "not the approved live handedness mapping",
                    "not the approved live mirror policy"),
                RawLandmarkObservationSet.Absent(33),
                RawLandmarkObservationSet.Absent(21),
                RawLandmarkObservationSet.Absent(21));

        static Step9CAmbiguousHandIdentityException Ambiguity(
            int providerFrameIndex,
            string reason = "pose_wrists_not_sufficiently_separated")
        {
            double timestamp = providerFrameIndex / 30.0;
            var diagnostic = new Step9CHandIdentityDiagnostic(
                providerFrameIndex: providerFrameIndex,
                providerTimestampSeconds: timestamp,
                handFrameTimestampSeconds: timestamp,
                sourceHandCount: 2,
                sourceCategory0: "Left",
                sourceScore0: 0.717995285987854,
                sourceCategory1: "Left",
                sourceScore1: 0.9925004243850708,
                hand0WristX: 0.25,
                hand0WristY: 0.50,
                hand1WristX: 0.75,
                hand1WristY: 0.50,
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
                reason: reason);
            return new Step9CAmbiguousHandIdentityException(diagnostic);
        }

        // -------------------------------------------------------------------------
        // The v1.0d3 pattern itself.
        // -------------------------------------------------------------------------

        [Test]
        public void SecondTypedAmbiguityNoLongerEndsTheCapture()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-second-typed"));
            source.Emit(Observation(1.0));

            source.Fail(Ambiguity(773));
            source.Fail(Ambiguity(774));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State,
                "the v1.0d3 live failure was exactly this second typed event");
            Assert.IsNull(controller.Failure);
            Assert.IsNull(controller.PrimaryFailure);
            Assert.AreEqual(1, controller.BufferedRawFrameCount,
                "the buffered observation must survive the drop");
            Assert.AreEqual(2, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.AreEqual(2, source.SubscriberCount,
                "the controller must remain subscribed to the source");
        }

        [Test]
        public void ManyTypedAmbiguitiesRemainNonfatalAndCountOncePerEvent()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-many-typed"));

            for (int ordinal = 1; ordinal <= 7; ordinal++)
            {
                source.Fail(Ambiguity(100 + ordinal));
                Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
                Assert.AreEqual(ordinal, controller.ToleratedAmbiguousHandIdentityCount,
                    "each typed event increments the capture drop count exactly once");
            }

            Assert.IsNull(controller.Failure);
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
        }

        [Test]
        public void EachTypedAmbiguityRefreshesTheLastDiagnostic()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-refresh"));
            source.Emit(Observation(1.0));

            source.Fail(Ambiguity(201, "reason_first"));
            PersonalLiveAmbiguityDiagnostic first =
                controller.LastToleratedAmbiguousHandIdentity;
            Assert.IsNotNull(first);
            StringAssert.Contains("reason_first", first.Message);
            Assert.AreEqual(1, first.BufferedRawFrameCount);
            Assert.IsTrue(first.DroppedDuringCapture);

            source.Emit(Observation(2.0));
            source.Fail(Ambiguity(202, "reason_second"));
            PersonalLiveAmbiguityDiagnostic second =
                controller.LastToleratedAmbiguousHandIdentity;
            Assert.IsNotNull(second);
            StringAssert.Contains("reason_second", second.Message);
            Assert.AreEqual(2, second.BufferedRawFrameCount,
                "the refreshed diagnostic reports the buffer as it stood at that event");
            Assert.AreEqual("v1-0e-refresh", second.CaptureSessionId);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, second.StateWhenRecorded);
            Assert.AreEqual(2, controller.ToleratedAmbiguousHandIdentityCount);
        }

        // -------------------------------------------------------------------------
        // No synthesis; valid frames keep their order, timestamps and geometry.
        // -------------------------------------------------------------------------

        [Test]
        public void ToleratedEventsNeverBufferOrSynthesizeAFrame()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-no-synthesis"));

            source.Fail(Ambiguity(301));
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
            source.Fail(Ambiguity(302));
            Assert.AreEqual(0, controller.BufferedRawFrameCount);

            source.Emit(Observation(1.0));
            Assert.AreEqual(1, controller.BufferedRawFrameCount);

            source.Fail(Ambiguity(303));
            Assert.AreEqual(1, controller.BufferedRawFrameCount,
                "a dropped event must not append, clone or duplicate an observation");
            Assert.AreEqual(3, controller.ToleratedAmbiguousHandIdentityCount);
        }

        [Test]
        public void ValidObservationsSurviveAroundDropsAndKeepStrictTimestampOrder()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-continuation"));

            source.Emit(Observation(1.0));
            source.Fail(Ambiguity(401));
            source.Emit(Observation(2.0));
            source.Fail(Ambiguity(402));
            source.Fail(Ambiguity(403));
            source.Emit(Observation(3.0));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.AreEqual(3, controller.BufferedRawFrameCount);
            Assert.AreEqual(3, controller.ToleratedAmbiguousHandIdentityCount);
        }

        [Test]
        public void ADropDoesNotAdvanceTheCaptureTimestampWatermark()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-watermark"));

            source.Emit(Observation(2.0));
            source.Fail(Ambiguity(501));

            // If the drop had moved the watermark, this stale observation could slip in.
            source.Emit(Observation(1.5));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.ObservationRejected,
                controller.Failure.Code,
                "strictly increasing timestamps are still enforced across a drop");
        }

        // -------------------------------------------------------------------------
        // Fail-closed and unchanged failure semantics.
        // -------------------------------------------------------------------------

        [Test]
        public void ACaptureOfOnlyDropsFailsEmptyCaptureAndNeverReachesRecognition()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-empty"));

            source.Fail(Ambiguity(601));
            source.Fail(Ambiguity(602));
            source.Fail(Ambiguity(603));
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);

            Assert.IsFalse(controller.FinalizeCapture());

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.EmptyCapture, controller.Failure.Code);
            Assert.AreEqual(PersonalLiveFailureCode.EmptyCapture,
                controller.PrimaryFailure.Code);
            Assert.IsNull(controller.Result);
            Assert.AreEqual(0, lastFactory.Runtime.RecognizeCallCount,
                "an ambiguity-only capture must never reach the model");
        }

        [Test]
        public void OptInFalseStillFailsFastOnTheFirstTypedAmbiguity()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, false);
            Assert.IsFalse(controller.CandidateTransientAmbiguityToleranceActive);
            Assert.IsTrue(controller.BeginCapture("v1-0e-opt-in-false"));
            source.Emit(Observation(1.0));

            source.Fail(Ambiguity(701));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            Assert.AreEqual(typeof(Step9CAmbiguousHandIdentityException).FullName,
                controller.PrimaryFailure.ExceptionType);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);
            Assert.AreEqual(0, source.SubscriberCount);
        }

        [Test]
        public void AGenericExceptionAfterDropsIsFatalAndPinsTheFirstFailure()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-generic"));
            source.Emit(Observation(1.0));
            source.Fail(Ambiguity(801));
            source.Fail(Ambiguity(802));

            source.Fail(new InvalidOperationException("injected generic bridge failure"));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            Assert.AreEqual(typeof(InvalidOperationException).FullName,
                controller.PrimaryFailure.ExceptionType);
            Assert.AreEqual(1, controller.PrimaryFailure.BufferedRawFrameCount);
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
            Assert.AreEqual(0, source.SubscriberCount);

            // A later typed event must not replace the preserved first fatal snapshot.
            source.Fail(Ambiguity(803));
            Assert.AreEqual(typeof(InvalidOperationException).FullName,
                controller.PrimaryFailure.ExceptionType);
            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
        }

        [Test]
        public void ANullObservationAfterDropsRemainsFatal()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-null-observation"));
            source.Fail(Ambiguity(901));

            source.Emit(null);

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.ObservationRejected,
                controller.Failure.Code);
        }

        [Test]
        public void AForeignConventionObservationAfterDropsRemainsFatal()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-foreign-convention"));
            source.Fail(Ambiguity(1001));

            source.Emit(ForeignConventionObservation(1.0));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.ObservationRejected,
                controller.Failure.Code);
        }

        // -------------------------------------------------------------------------
        // Lifecycle boundaries and terminal states.
        // -------------------------------------------------------------------------

        [Test]
        public void ResetClearsTheDropCountAndTheStaleDiagnosticForTheNextCapture()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-before-reset"));
            source.Fail(Ambiguity(1101));
            source.Fail(Ambiguity(1102));
            Assert.AreEqual(2, controller.ToleratedAmbiguousHandIdentityCount);

            Assert.IsFalse(controller.FinalizeCapture(),
                "a capture of only drops cannot finalize");
            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);

            Assert.IsTrue(controller.Reset());
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);
            Assert.IsNull(controller.PrimaryFailure);

            Assert.IsTrue(controller.BeginCapture("v1-0e-after-reset"));
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);

            source.Fail(Ambiguity(1103));
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.AreEqual(1, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.AreEqual("v1-0e-after-reset", controller.CaptureSessionId);
        }

        [Test]
        public void BeginCaptureClearsADropCountLeftByThePreviousCapture()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("v1-0e-first-capture"));
            source.Fail(Ambiguity(1201));
            source.Fail(Ambiguity(1202));
            source.Fail(Ambiguity(1203));
            Assert.AreEqual(3, controller.ToleratedAmbiguousHandIdentityCount);

            Assert.IsTrue(controller.CancelCapture());
            Assert.IsTrue(controller.Reset());
            Assert.IsTrue(controller.BeginCapture("v1-0e-second-capture"));

            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);
        }

        [Test]
        public void ATypedAmbiguityWhileReadyStaysDiagnosticOnlyAndDoesNotCount()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);

            source.Fail(Ambiguity(1301));

            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.IsNull(controller.Failure);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount,
                "a Ready-state ambiguity is not a capture drop");
            Assert.IsNotNull(controller.LastToleratedAmbiguousHandIdentity);
            Assert.IsFalse(
                controller.LastToleratedAmbiguousHandIdentity.DroppedDuringCapture);
        }

        [Test]
        public void TerminalStatesIgnoreLaterTypedAmbiguityEvents()
        {
            var cancelledSource = new FakeObservationSource();
            PersonalLiveRecognitionController cancelled =
                ReadyController(cancelledSource, true);
            Assert.IsTrue(cancelled.BeginCapture("v1-0e-cancelled"));
            cancelledSource.Fail(Ambiguity(1401));
            Assert.IsTrue(cancelled.CancelCapture());

            cancelledSource.Fail(Ambiguity(1402));

            Assert.AreEqual(PersonalLiveCaptureState.Cancelled, cancelled.State);
            Assert.AreEqual(1, cancelled.ToleratedAmbiguousHandIdentityCount,
                "a terminal state must not process a later typed event");
            Assert.IsNull(cancelled.Failure);

            var failedSource = new FakeObservationSource();
            PersonalLiveRecognitionController failed = ReadyController(failedSource, true);
            Assert.IsTrue(failed.BeginCapture("v1-0e-failed"));
            failedSource.Fail(new InvalidOperationException("injected generic failure"));
            Assert.AreEqual(PersonalLiveCaptureState.Failed, failed.State);

            failedSource.Fail(Ambiguity(1403));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, failed.State);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, failed.Failure.Code);
            Assert.AreEqual(0, failed.ToleratedAmbiguousHandIdentityCount);
        }
    }
}
