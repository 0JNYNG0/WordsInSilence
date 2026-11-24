// F7R1 EditMode tests for the repeat coordinator. Pure and synthetic: no scene, camera,
// prefab, model, real capture or network. Nothing here has been executed.
using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Recognition.FullHandRepeat;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandRepeatCoordinatorTests
    {
        /// <summary>
        /// EnsureControllerReady is DESTRUCTIVE here, exactly as the real harness is when
        /// State is not Ready: it rebuilds the controller, issues a new id and clears the
        /// failure. Any readiness tick that reaches it during or after a take is visible.
        /// </summary>
        sealed class DestructiveFakePort : IFullHandRepeatHarnessPort
        {
            public bool Bound = true;
            public bool SelectionAccepted = true;
            public bool SelectionEffective = true;
            public bool StartAccepted = true;
            public bool ThrowOnStart;
            public bool FinalizeAccepted = true;
            public bool ThrowOnFinalize;
            public string Effective = string.Empty;
            public PersonalLiveCaptureState CurrentState = PersonalLiveCaptureState.Idle;
            public int Buffered = 24;
            public string Session = string.Empty;
            public string Failure = string.Empty;
            public int Rebuilds, StartCalls, FinalizeCalls, CancelCalls, ResetCalls;
            int sessionCounter;

            public bool LiveInputBound { get { return Bound; } }
            public string EffectiveTarget { get { return Effective; } }
            public PersonalLiveCaptureState State { get { return CurrentState; } }
            public string CaptureSessionId { get { return Session; } }
            public int BufferedRawFrameCount { get { return Buffered; } }

            public bool EnsureControllerReady()
            {
                if (CurrentState != PersonalLiveCaptureState.Ready)
                {
                    Rebuilds++;
                    Session = string.Empty;         // a rebuild destroys the capture id
                    Failure = string.Empty;         // and the recorded early failure
                    CurrentState = PersonalLiveCaptureState.Ready;
                }
                return CurrentState == PersonalLiveCaptureState.Ready;
            }

            public bool TrySelectTarget(string word)
            {
                if (!SelectionAccepted) return false;
                if (SelectionEffective) Effective = word;
                return true;
            }

            public bool StartCapture()
            {
                StartCalls++;
                if (ThrowOnStart) throw new InvalidOperationException("injected start");
                if (!StartAccepted) return false;
                sessionCounter++;
                Session = "capture_" + sessionCounter;
                CurrentState = PersonalLiveCaptureState.Capturing;
                return true;
            }

            public bool FinalizeCapture()
            {
                FinalizeCalls++;
                if (ThrowOnFinalize) throw new InvalidOperationException("injected finalize");
                CurrentState = FinalizeAccepted ? PersonalLiveCaptureState.Completed
                                                : PersonalLiveCaptureState.Failed;
                return FinalizeAccepted;
            }

            public bool CancelCapture()
            {
                CancelCalls++;
                CurrentState = PersonalLiveCaptureState.Cancelled;
                return true;
            }

            public bool ResetCapture()
            {
                ResetCalls++;
                Session = string.Empty;
                CurrentState = PersonalLiveCaptureState.Ready;
                return true;
            }

            /// <summary>Simulates the SourceFailed route failing this capture.</summary>
            public void FailFromSource()
            {
                CurrentState = PersonalLiveCaptureState.Failed;
                Failure = "BridgeFailure";
                Buffered = 0;                        // the controller clears its buffer
            }
        }

        sealed class RecordingSink : IFullHandObservationSink
        {
            public readonly Dictionary<string, string> Text =
                new Dictionary<string, string>(StringComparer.Ordinal);
            public bool ThrowOnWrite;

            public void WriteBytes(string name, byte[] payload)
            {
                if (ThrowOnWrite) throw new System.IO.IOException("injected");
            }

            public void WriteText(string name, string text)
            {
                if (ThrowOnWrite) throw new System.IO.IOException("injected");
                if (Text.ContainsKey(name))
                    throw new System.IO.IOException("CreateNew refuses to overwrite " + name);
                Text[name] = text;
            }
        }

        sealed class StepClock
        {
            DateTime now = new DateTime(2026, 9, 9, 12, 0, 0, DateTimeKind.Utc);
            public DateTime Next() { now = now.AddSeconds(2.5); return now; }
        }

        static string FirstWord()
        {
            return PersonalLiveRecognitionController.SupportedWords[0];
        }

        static FullHandRepeatCoordinator Build(IFullHandRepeatHarnessPort port,
                                               RecordingSink sink, Func<DateTime> clock = null)
        {
            return new FullHandRepeatCoordinator(port, sink, line => { }, clock);
        }

        // ---- 1. readiness lifecycle ------------------------------------------

        [Test]
        public void ReadinessTicksNeverDestroyAnActiveOrTerminalController()
        {
            var port = new DestructiveFakePort();
            var sink = new RecordingSink();
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            string reason;

            Assert.IsTrue(coordinator.EnsureReadyForNextTake());
            Assert.AreEqual(1, port.Rebuilds, "readiness is prepared exactly once");
            for (int tick = 0; tick < 10; tick++) coordinator.EnsureReadyForNextTake();
            Assert.AreEqual(1, port.Rebuilds, "a ready controller is never rebuilt again");

            Assert.IsTrue(coordinator.TryBegin(FirstWord(), "positive", "development",
                                               out reason), reason);
            string capturedId = coordinator.ActiveCaptureSessionId;
            Assert.IsNotEmpty(capturedId);
            for (int tick = 0; tick < 20; tick++) coordinator.EnsureReadyForNextTake();
            Assert.AreEqual(1, port.Rebuilds, "no rebuild while capturing");
            Assert.AreEqual(capturedId, port.Session, "the capture id survives every tick");

            // the source fails this capture; the early failure must survive status ticks
            port.FailFromSource();
            for (int tick = 0; tick < 20; tick++) coordinator.EnsureReadyForNextTake();
            Assert.AreEqual(1, port.Rebuilds, "a terminal controller is never rebuilt");
            Assert.AreEqual("BridgeFailure", port.Failure, "the early failure is preserved");

            Assert.IsFalse(coordinator.TryFinish(false, out reason));
            Assert.AreEqual(0, port.FinalizeCalls,
                            "an already failed capture is not finalized over");
            string record = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("\"outcome\": \"already_failed_before_close\"", record);
            StringAssert.Contains("\"buffered_raw_frame_count\": null", record);
            StringAssert.Contains("\"buffered_raw_frame_count_available\": false", record);

            // only an explicit reset, after the outcome was preserved, rebuilds readiness
            Assert.IsTrue(coordinator.TryPrepareNext(out reason), reason);
            Assert.AreEqual(1, port.ResetCalls);
        }

        // ---- 5. start refusal is a recorded take -------------------------------

        [Test]
        public void ARefusedOrThrowingStartIsRecordedAsAFailedRequestedTake()
        {
            var refusing = new DestructiveFakePort { StartAccepted = false };
            var sink = new RecordingSink();
            FullHandRepeatCoordinator coordinator = Build(refusing, sink);
            string reason;

            Assert.IsFalse(coordinator.TryBegin(FirstWord(), "positive", "development",
                                                out reason));
            Assert.AreEqual("START_REFUSED", reason);
            Assert.IsFalse(coordinator.CaptureActive);
            Assert.AreEqual(1, coordinator.AttemptCount);
            string record = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("\"outcome\": \"START_REFUSED\"", record);
            StringAssert.Contains("\"private_capture_session_id\": null", record);
            StringAssert.Contains("\"capture_session_id_present\": false", record);

            var throwing = new DestructiveFakePort { ThrowOnStart = true };
            var secondSink = new RecordingSink();
            FullHandRepeatCoordinator second = Build(throwing, secondSink);
            Assert.IsFalse(second.TryBegin(FirstWord(), "idle", "final", out reason));
            StringAssert.StartsWith("START_ERROR_", reason);
            StringAssert.Contains("START_ERROR_InvalidOperationException",
                                  secondSink.Text["repeat_attempt_0001.json"]);
        }

        [Test]
        public void TheSequenceNumberIsReservedBeforeTheStartAttempt()
        {
            var port = new DestructiveFakePort { StartAccepted = false };
            var sink = new RecordingSink();
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            string reason;

            coordinator.TryBegin(FirstWord(), "positive", "development", out reason);
            port.StartAccepted = true;
            coordinator.TryBegin(FirstWord(), "positive", "development", out reason);
            coordinator.TryFinish(false, out reason);

            Assert.IsTrue(sink.Text.ContainsKey("repeat_attempt_0001.json"));
            Assert.IsTrue(sink.Text.ContainsKey("repeat_attempt_0002.json"),
                          "a consumed number is never reused, so nothing is overwritten");
            Assert.AreEqual(2, sink.Text.Count);
        }

        // ---- 5. cancel semantics and timestamps ---------------------------------

        [Test]
        public void CancelReportsASuccessfulRecordingAndFinishReportsTheCapture()
        {
            var port = new DestructiveFakePort();
            var sink = new RecordingSink();
            var clock = new StepClock();
            FullHandRepeatCoordinator coordinator = Build(port, sink, clock.Next);
            string reason;

            coordinator.TryBegin(FirstWord(), "positive", "development", out reason);
            Assert.IsTrue(coordinator.TryCancel(out reason),
                          "a deliberate cancel that was recorded reports true");
            Assert.AreEqual(1, port.CancelCalls);
            string cancelled = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("\"outcome\": \"cancelled\"", cancelled);
            StringAssert.Contains("\"start_utc\": \"2026-09-09T12:00:02.500Z\"", cancelled);
            StringAssert.Contains("\"end_utc\": \"2026-09-09T12:00:05.000Z\"", cancelled);
            StringAssert.Contains("\"elapsed_seconds\": 2.5", cancelled);

            Assert.IsTrue(coordinator.TryPrepareNext(out reason), reason);
            Assert.IsTrue(coordinator.TryBegin(FirstWord(), "positive", "development",
                                               out reason), reason);
            Assert.IsTrue(coordinator.TryFinish(false, out reason), reason);
            StringAssert.Contains("\"outcome\": \"completed\"",
                                  sink.Text["repeat_attempt_0002.json"]);
        }

        // ---- 5/6. labels and the absence of score data --------------------------

        [Test]
        public void LabelsAreRequestedAndUnconfirmedAndNoScoreDataIsStored()
        {
            var port = new DestructiveFakePort();
            var sink = new RecordingSink();
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            string reason;

            coordinator.TryBegin(FirstWord(), "wrong_handshape", "final", out reason);
            coordinator.TryFinish(false, out reason);

            string record = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("\"intended_ground_truth\": \""
                                  + FullHandRepeatContract.NoValidSign + "\"", record);
            StringAssert.Contains("\"label_status\": \"requested_unconfirmed\"", record);
            StringAssert.Contains("\"harness_api_target\": \"" + FirstWord() + "\"", record);
            StringAssert.Contains("\"harness_api_target_is_intended_ground_truth\": false",
                                  record);
            StringAssert.Contains("\"partition\": \"final\"", record);
            // no score, acceptance or correctness FIELD may exist in this record
            foreach (string field in new[] { "\"score\":", "\"scores\":", "\"accepted\":",
                                             "\"predicted\":", "\"distance\":",
                                             "\"correct\":", "\"quest_success\":" })
                StringAssert.DoesNotContain(field, record);
        }

        [Test]
        public void APositiveConditionKeepsTheRequestedWordAsTheIntendedGroundTruth()
        {
            var port = new DestructiveFakePort();
            var sink = new RecordingSink();
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            string reason;

            coordinator.TryBegin(FirstWord(), "positive", "development", out reason);
            coordinator.TryFinish(true, out reason);

            string record = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("\"intended_ground_truth\": \"" + FirstWord() + "\"", record);
            StringAssert.Contains("\"harness_api_target_is_intended_ground_truth\": true",
                                  record);
            StringAssert.Contains("\"auto_finished_at_duration_bound\": true", record);
            StringAssert.Contains("\"buffered_raw_frame_count\": 24", record);
            StringAssert.Contains("trailing trim of 0 seconds", record);
        }

        // ---- configuration and stale selection ----------------------------------

        [Test]
        public void AnInvalidConfigurationOrStaleSelectionNeverStartsOrRecords()
        {
            var sink = new RecordingSink();
            string reason;

            var port = new DestructiveFakePort();
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            Assert.IsFalse(coordinator.TryBegin("NOT_A_WORD", "positive", "development",
                                                out reason));
            Assert.AreEqual("TARGET_NOT_SUPPORTED", reason);
            Assert.IsFalse(coordinator.TryBegin(FirstWord(), "nope", "development", out reason));
            Assert.AreEqual("CONDITION_INVALID", reason);
            Assert.IsFalse(coordinator.TryBegin(FirstWord(), "positive", "nope", out reason));
            Assert.AreEqual("PARTITION_INVALID", reason);

            var unbound = new DestructiveFakePort { Bound = false };
            Assert.IsFalse(Build(unbound, sink).TryBegin(FirstWord(), "positive",
                                                          "development", out reason));
            Assert.AreEqual("LIVE_INPUT_NOT_BOUND", reason);

            var stale = new DestructiveFakePort { SelectionEffective = false,
                                                  Effective = "STALE_WORD" };
            Assert.IsFalse(Build(stale, sink).TryBegin(FirstWord(), "positive", "development",
                                                        out reason));
            Assert.AreEqual("TARGET_SELECTION_NOT_EFFECTIVE", reason);

            Assert.AreEqual(0, port.StartCalls);
            Assert.AreEqual(0, sink.Text.Count, "a refused configuration records nothing");
        }

        // ---- faults ---------------------------------------------------------------

        [Test]
        public void ARecordingFaultIsVisibleAndNeverGrantsSuccess()
        {
            var port = new DestructiveFakePort();
            var sink = new RecordingSink { ThrowOnWrite = true };
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            string reason;

            coordinator.TryBegin(FirstWord(), "positive", "development", out reason);
            Assert.IsFalse(coordinator.TryFinish(false, out reason));
            Assert.AreEqual("RECORDING_FAULT", reason);
            Assert.AreEqual(1, coordinator.RecordingFaultCount);
            Assert.AreEqual(0, coordinator.RecordedCount);
            Assert.IsFalse(coordinator.CaptureActive);
        }

        [Test]
        public void HarnessAndLogFaultsAreContainedAndTheTakeIsStillRecorded()
        {
            var port = new DestructiveFakePort { ThrowOnFinalize = true };
            var sink = new RecordingSink();
            var coordinator = new FullHandRepeatCoordinator(
                port, sink, line => { throw new InvalidOperationException("injected"); }, null);
            string reason = string.Empty;

            Assert.DoesNotThrow(delegate
            {
                coordinator.TryBegin(FirstWord(), "positive", "development", out reason);
                coordinator.TryFinish(false, out reason);
            });
            Assert.IsFalse(coordinator.CaptureActive);
            string record = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("HARNESS_ERROR_InvalidOperationException", record);
            StringAssert.Contains("\"outcome\": \"failed\"", record);
        }

        [Test]
        public void ClosingWithoutAnActiveTakeIsRefusedAndRecordsNothing()
        {
            var port = new DestructiveFakePort();
            var sink = new RecordingSink();
            FullHandRepeatCoordinator coordinator = Build(port, sink);
            string reason;

            Assert.IsFalse(coordinator.TryFinish(false, out reason));
            Assert.AreEqual("NO_ACTIVE_CAPTURE", reason);
            Assert.IsFalse(coordinator.TryCancel(out reason));
            Assert.AreEqual("NO_ACTIVE_CAPTURE", reason);
            Assert.AreEqual(0, sink.Text.Count);
            Assert.AreEqual(0, port.FinalizeCalls);
            Assert.AreEqual(0, port.CancelCalls);
        }
    }
}
