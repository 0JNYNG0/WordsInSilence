using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class CandidateTransientAmbiguousHandIdentityToleranceV1Tests
    {
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

            public void Emit(RawMediaPipeObservation observation) => observations?.Invoke(observation);
            public void Fail(Exception exception) => failures?.Invoke(exception);
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime
        {
        }

        sealed class FakeRuntime : IPersonalLiveRecognitionRuntime
        {
            public bool IsPersonalActive => true;
            public string ActiveProfileId => "step9p-candidate-test";

            public PersonalLiveRoutingResult Recognize(
                string sessionId,
                RecognitionSequence sequence,
                string expectedWord)
            {
                throw new NotSupportedException("recognition is outside this tolerance fixture");
            }
        }

        sealed class FakeRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            readonly FakeRuntime runtime = new FakeRuntime();

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) =>
                new FakeGlobalRuntime();

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime,
                PersonalLiveRuntimePaths paths) => runtime;
        }

        [Test]
        public void LegacyController_AmbiguousHandIdentityRemainsFailFast()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, false);

            source.Fail(Ambiguity(10));

            Assert.IsFalse(controller.CandidateTransientAmbiguityToleranceActive);
            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            Assert.AreEqual(typeof(Step9CAmbiguousHandIdentityException).FullName,
                controller.PrimaryFailure.ExceptionType);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);
            Assert.AreEqual(0, source.SubscriberCount);
        }

        [Test]
        public void CandidateReady_AmbiguityRecordsDiagnosticWithoutChangingState()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);

            source.Fail(Ambiguity(20));

            Assert.IsTrue(controller.CandidateTransientAmbiguityToleranceActive);
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.IsNull(controller.Failure);
            Assert.IsNull(controller.PrimaryFailure);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.AreEqual(2, source.SubscriberCount);
            PersonalLiveAmbiguityDiagnostic diagnostic =
                controller.LastToleratedAmbiguousHandIdentity;
            Assert.IsNotNull(diagnostic);
            Assert.AreEqual(PersonalLiveCaptureState.Ready, diagnostic.StateWhenRecorded);
            Assert.AreEqual(string.Empty, diagnostic.CaptureSessionId);
            Assert.AreEqual(0, diagnostic.BufferedRawFrameCount);
            Assert.IsFalse(diagnostic.DroppedDuringCapture);
            Assert.AreEqual(typeof(Step9CAmbiguousHandIdentityException).FullName,
                diagnostic.ExceptionType);
        }

        [Test]
        public void CandidateCapturing_FirstAmbiguityPreservesSessionBufferAndAcceptsNextObservation()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("candidate-first-drop"));
            source.Emit(Observation(1.0));

            source.Fail(Ambiguity(30));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.AreEqual("candidate-first-drop", controller.CaptureSessionId);
            Assert.AreEqual(1, controller.BufferedRawFrameCount);
            Assert.AreEqual(1, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.Failure);
            Assert.IsNull(controller.PrimaryFailure);
            Assert.AreEqual(2, source.SubscriberCount);
            PersonalLiveAmbiguityDiagnostic diagnostic =
                controller.LastToleratedAmbiguousHandIdentity;
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, diagnostic.StateWhenRecorded);
            Assert.AreEqual("candidate-first-drop", diagnostic.CaptureSessionId);
            Assert.AreEqual(1, diagnostic.BufferedRawFrameCount);
            Assert.IsTrue(diagnostic.DroppedDuringCapture);

            source.Emit(Observation(2.0));
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.AreEqual(2, controller.BufferedRawFrameCount);
        }

        /// <summary>
        /// Step 10P v1.0e1: a typed ambiguity is a frame the resolver could not attribute
        /// to a hand, so the approved response is to drop it and keep capturing. The
        /// v1.0d3 live run failed because the second such event exhausted a one-event
        /// tolerance and was promoted to a fatal BridgeFailure, discarding the frames the
        /// capture had already collected. This test now pins the opposite: every typed
        /// ambiguity is survivable, and nothing is invented in the dropped frame's place.
        /// </summary>
        [Test]
        public void CandidateCapturing_MultipleAmbiguitiesRemainNonfatalAndPreserveBuffer()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("candidate-multi-drop"));
            source.Emit(Observation(1.0));
            Assert.AreEqual(1, controller.BufferedRawFrameCount);

            source.Fail(Ambiguity(40));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.IsNull(controller.Failure);
            Assert.IsNull(controller.PrimaryFailure);
            Assert.AreEqual(1, controller.BufferedRawFrameCount);
            Assert.AreEqual(1, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.AreEqual(2, source.SubscriberCount);

            source.Fail(Ambiguity(41));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.IsNull(controller.Failure);
            Assert.IsNull(controller.PrimaryFailure);
            Assert.AreEqual(1, controller.BufferedRawFrameCount);
            Assert.AreEqual(2, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.AreEqual(2, source.SubscriberCount);
            Assert.AreEqual("candidate-multi-drop", controller.CaptureSessionId);

            PersonalLiveAmbiguityDiagnostic diagnostic =
                controller.LastToleratedAmbiguousHandIdentity;
            Assert.IsNotNull(diagnostic);
            Assert.IsTrue(diagnostic.DroppedDuringCapture);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, diagnostic.StateWhenRecorded);
            Assert.AreEqual("candidate-multi-drop", diagnostic.CaptureSessionId);
            Assert.AreEqual(1, diagnostic.BufferedRawFrameCount);
            Assert.AreEqual(typeof(Step9CAmbiguousHandIdentityException).FullName,
                diagnostic.ExceptionType);

            // The diagnostic describes the SECOND event, not the first.
            StringAssert.Contains("provider frame 41", diagnostic.Message);
            Assert.IsFalse(diagnostic.Message.Contains("provider frame 40"));

            // Both duplicate provider categories are recorded verbatim: the controller
            // reports the ambiguity rather than picking a hand identity for it.
            StringAssert.Contains("categories=Left(0.91),Left(0.99)", diagnostic.Message);

            // Neither drop adopted a timestamp. Ambiguity(40) and Ambiguity(41) carry
            // timestamps of 40/30 and 41/30 seconds; if either had advanced the capture
            // watermark, this 1.1 s observation would be rejected as non-increasing.
            source.Emit(Observation(1.1));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.IsNull(controller.Failure);
            Assert.AreEqual(2, controller.BufferedRawFrameCount);
            Assert.AreEqual(2, controller.ToleratedAmbiguousHandIdentityCount);
        }

        [Test]
        public void CandidateCapturing_GenericBridgeExceptionStillFailsImmediately()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("candidate-generic-fails"));
            source.Emit(Observation(1.0));

            source.Fail(new InvalidOperationException("injected generic bridge failure"));

            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            Assert.AreEqual(typeof(InvalidOperationException).FullName,
                controller.PrimaryFailure.ExceptionType);
            Assert.AreEqual(1, controller.PrimaryFailure.BufferedRawFrameCount);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);
            Assert.AreEqual(0, source.SubscriberCount);
        }

        /// <summary>
        /// Step 10P v1.0e1: this test used to reach Reset by letting the second typed
        /// ambiguity fail the capture. That route no longer exists, and manufacturing a
        /// failure to recreate it would test the fixture rather than the controller. It
        /// now reaches a terminal state the production lifecycle actually offers -
        /// CancelCapture - and proves the reset clears the ambiguity state so the next
        /// capture starts clean.
        /// </summary>
        [Test]
        public void CandidateResetAfterCancelledCapture_ClearsAmbiguityStateForNextCapture()
        {
            var source = new FakeObservationSource();
            PersonalLiveRecognitionController controller = ReadyController(source, true);
            Assert.IsTrue(controller.BeginCapture("candidate-before-reset"));
            source.Fail(Ambiguity(50));
            source.Fail(Ambiguity(51));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.AreEqual(2, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.Failure);
            Assert.IsNotNull(controller.LastToleratedAmbiguousHandIdentity);

            Assert.IsTrue(controller.CancelCapture());
            Assert.AreEqual(PersonalLiveCaptureState.Cancelled, controller.State);

            Assert.IsTrue(controller.Reset());
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);
            Assert.IsNull(controller.Failure);
            Assert.IsNull(controller.PrimaryFailure);

            Assert.IsTrue(controller.BeginCapture("candidate-after-reset"));
            Assert.AreEqual(0, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.LastToleratedAmbiguousHandIdentity);

            source.Fail(Ambiguity(52));

            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            Assert.AreEqual("candidate-after-reset", controller.CaptureSessionId);
            Assert.AreEqual(1, controller.ToleratedAmbiguousHandIdentityCount);
            Assert.IsNull(controller.Failure);
            Assert.AreEqual(2, source.SubscriberCount);
        }

        [Test]
        public void CandidateHarnessOnDisable_RemovesAllSubscribersAndCancelsController()
        {
            GameObject owner = null;
            try
            {
                var source = new FakeObservationSource();
                PersonalLiveRecognitionController controller = ReadyController(source, true);
                owner = new GameObject("Step9P-Candidate-OnDisable");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness =
                    owner.AddComponent<PersonalLiveRecognitionHarness>();
                SetPrivateField(harness, "profileMode",
                    PersonalLiveProfileMode.MultiPrototypeCandidate);
                SetPrivateField(harness, "controller", controller);

                Assert.AreEqual(PersonalLiveProfileMode.MultiPrototypeCandidate, harness.ProfileMode);
                Assert.AreEqual(2, source.SubscriberCount);
                InvokeLifecycle(harness, "OnDisable");
                Assert.AreEqual(0, source.SubscriberCount);
                Assert.AreEqual(PersonalLiveCaptureState.Cancelled, controller.State);
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        [Test]
        public void CandidateHarnessOnDestroy_RemovesAllSubscribersAndClearsControllerReference()
        {
            GameObject owner = null;
            try
            {
                var source = new FakeObservationSource();
                PersonalLiveRecognitionController controller = ReadyController(source, true);
                owner = new GameObject("Step9P-Candidate-OnDestroy");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness =
                    owner.AddComponent<PersonalLiveRecognitionHarness>();
                SetPrivateField(harness, "profileMode",
                    PersonalLiveProfileMode.MultiPrototypeCandidate);
                SetPrivateField(harness, "controller", controller);

                Assert.AreEqual(2, source.SubscriberCount);
                InvokeLifecycle(harness, "OnDestroy");
                Assert.AreEqual(0, source.SubscriberCount);
                Assert.IsNull(GetPrivateField(harness, "controller"));
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        static PersonalLiveRecognitionController ReadyController(
            FakeObservationSource source,
            bool candidateTolerance)
        {
            var controller = new PersonalLiveRecognitionController(
                source,
                new FakeRuntimeFactory(),
                enableCandidateTransientAmbiguityTolerance: candidateTolerance);
            Assert.IsTrue(controller.Initialize(new PersonalLiveRuntimePaths(
                "global-templates",
                "global-threshold",
                "personal-profile")));
            return controller;
        }

        static RawMediaPipeObservation Observation(double timestamp) =>
            new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.LiveMediaPipe,
                RawLandmarkObservationSet.Absent(33),
                RawLandmarkObservationSet.Absent(21),
                RawLandmarkObservationSet.Absent(21));

        static Step9CAmbiguousHandIdentityException Ambiguity(int providerFrameIndex)
        {
            double timestamp = providerFrameIndex / 30.0;
            var diagnostic = new Step9CHandIdentityDiagnostic(
                providerFrameIndex: providerFrameIndex,
                providerTimestampSeconds: timestamp,
                handFrameTimestampSeconds: timestamp,
                sourceHandCount: 2,
                sourceCategory0: "Left",
                sourceScore0: 0.91,
                sourceCategory1: "Left",
                sourceScore1: 0.99,
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
                reason: "same_frame_pose_wrists_unavailable_or_low_visibility");
            return new Step9CAmbiguousHandIdentityException(diagnostic);
        }

        static void SetPrivateField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, fieldName);
            field.SetValue(target, value);
        }

        static object GetPrivateField(object target, string fieldName)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, fieldName);
            return field.GetValue(target);
        }

        static void InvokeLifecycle(PersonalLiveRecognitionHarness harness, string methodName)
        {
            MethodInfo method = typeof(PersonalLiveRecognitionHarness).GetMethod(
                methodName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(method, methodName);
            method.Invoke(harness, null);
        }
    }
}
