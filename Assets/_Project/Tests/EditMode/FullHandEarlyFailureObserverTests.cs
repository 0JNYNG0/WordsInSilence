// F6R1 EditMode tests for the early-failure observer. Synthetic only: no webcam, no real
// capture, no model, no profile content, no network. Filesystem cases use an isolated temp
// directory. The controller cases drive the REAL SourceFailed event route with the minimal
// fake source/runtime pattern from PersonalLiveRecognitionIntegrationV1Tests.
// Nothing here has been executed.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using NUnit.Framework;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandEarlyFailureObserverTests
    {
        sealed class RecordingSink : IFullHandObservationSink
        {
            public readonly Dictionary<string, string> Text =
                new Dictionary<string, string>(StringComparer.Ordinal);
            public bool ThrowOnWrite;

            public void WriteBytes(string name, byte[] payload)
            {
                if (ThrowOnWrite) throw new IOException("injected");
            }

            public void WriteText(string name, string text)
            {
                if (ThrowOnWrite) throw new IOException("injected");
                if (Text.ContainsKey(name))
                    throw new IOException("CreateNew would refuse to overwrite " + name);
                Text[name] = text;
            }
        }

        [SetUp]
        public void SetUp() { FullHandEarlyFailureObserver.ResetForTest(); }

        [TearDown]
        public void TearDown() { FullHandEarlyFailureObserver.ResetForTest(); }

        static void Report(string session, string target, string stage, string code,
                           string exceptionType, int count)
        {
            FullHandEarlyFailureObserver.SafeReportFailure(session, target, stage, code,
                                                           exceptionType, count);
        }

        static RecordingSink ActiveSink()
        {
            var sink = new RecordingSink();
            FullHandEarlyFailureObserver.SinkForTest = sink;
            FullHandEarlyFailureObserver.LogForTest = line => { };
            return sink;
        }

        // ---- disabled ---------------------------------------------------------

        [Test]
        public void DisabledRequestsOnlyTheEnableFlagAndNeverLogs()
        {
            var requested = new List<string>();
            int consoleCalls = 0;
            FullHandEarlyFailureObserver.EnvironmentForTest =
                name => { requested.Add(name); return null; };
            FullHandEarlyFailureObserver.LogForTest = line => { consoleCalls++; };

            Assert.DoesNotThrow(delegate
            {
                Report("session_a", "GREETING", "Capturing", "BridgeFailure", "IOException", 12);
                Report("session_b", "GREETING", "Capturing", "BridgeFailure", "IOException", 3);
            });

            // the output root name is never even asked for, so no directory can be touched
            CollectionAssert.AreEqual(new[] { FullHandAttemptObserver.EnableVariable },
                                      requested);
            Assert.AreEqual(0, consoleCalls, "a disabled observer writes no console line");
            Assert.AreEqual(FullHandObserverState.Disabled, FullHandEarlyFailureObserver.State);
            Assert.AreEqual("NOT_ENABLED", FullHandEarlyFailureObserver.ReasonCode);
            Assert.AreEqual(0, FullHandEarlyFailureObserver.RecordCount);
            Assert.AreEqual(0, FullHandEarlyFailureObserver.RememberedKeyCount);
        }

        // ---- configuration refusal ---------------------------------------------

        [Test]
        public void RelativeMissingAndInAssetsRootsAreRefusedWithoutThrowing()
        {
            string temporary = Path.Combine(Path.GetTempPath(),
                "wis_f6r1_" + Guid.NewGuid().ToString("N"));
            string inAssets = Path.Combine(temporary, "Assets", "observations");
            Directory.CreateDirectory(inAssets);
            try
            {
                var cases = new List<KeyValuePair<string, string>>
                {
                    new KeyValuePair<string, string>("relative_root", "OUTPUT_ROOT_NOT_ABSOLUTE"),
                    new KeyValuePair<string, string>("C:relative_root", "OUTPUT_ROOT_NOT_ABSOLUTE"),
                    new KeyValuePair<string, string>(Path.Combine(temporary, "absent"),
                                                     "OBSERVATION_ROOT_MISSING"),
                    new KeyValuePair<string, string>(inAssets, "OBSERVATION_ROOT_REFUSED"),
                };
                foreach (KeyValuePair<string, string> entry in cases)
                {
                    FullHandEarlyFailureObserver.ResetForTest();
                    var lines = new List<string>();
                    FullHandEarlyFailureObserver.LogForTest = lines.Add;
                    FullHandEarlyFailureObserver.EnvironmentForTest = name =>
                    {
                        if (name == FullHandAttemptObserver.EnableVariable) return "1";
                        if (name == FullHandAttemptObserver.OutputVariable) return entry.Key;
                        return null;
                    };

                    Assert.DoesNotThrow(delegate
                    {
                        Report("session_b", "GREETING", "Ready", "BridgeMissing", null, 0);
                    });
                    Assert.AreEqual(FullHandObserverState.Faulted,
                                    FullHandEarlyFailureObserver.State, entry.Key);
                    Assert.AreEqual(entry.Value, FullHandEarlyFailureObserver.ReasonCode,
                                    entry.Key);
                    Assert.AreEqual(0, FullHandEarlyFailureObserver.RecordCount);
                    Assert.AreEqual(1, lines.Count, "one short setup line only");
                    StringAssert.DoesNotContain("session_b", lines[0]);
                    StringAssert.DoesNotContain(entry.Key, lines[0]);
                }
            }
            finally
            {
                try { Directory.Delete(temporary, true); } catch (Exception) { }
            }
        }

        // ---- record content -----------------------------------------------------

        [Test]
        public void TheRecordScopesEveryClaimToThisDiagnosticOnly()
        {
            RecordingSink sink = ActiveSink();

            Report("capture_17", "GREETING", "Processing", "RecognitionFailure",
                   "InvalidDataException", 87);

            string record = sink.Text["early_failure_0001.json"];
            StringAssert.Contains("\"schema\": \"wis_full_hand_early_failure_v1\"", record);
            StringAssert.Contains("\"stage\": \"Processing\"", record);
            StringAssert.Contains("\"failure_code\": \"RecognitionFailure\"", record);
            StringAssert.Contains("\"exception_type\": \"InvalidDataException\"", record);
            StringAssert.Contains("\"buffered_observation_count\": 87", record);
            StringAssert.Contains("\"buffered_observation_count_available\": true", record);
            StringAssert.Contains("\"observer_loaded_model\": false", record);
            StringAssert.Contains("\"observer_scored\": false", record);
            StringAssert.Contains("\"attempt_model_and_snapshot_state\": "
                                  + "\"unobserved_by_this_diagnostic\"", record);
            // it must not assert anything about the attempt as a whole
            StringAssert.DoesNotContain("\"model_loaded\"", record);
            StringAssert.DoesNotContain("\"scored\": false", record);
            StringAssert.DoesNotContain("before any recognition snapshot existed", record);
            StringAssert.Contains("UNOBSERVED here, not false", record);
        }

        [Test]
        public void AbsentMetadataStaysNullAndANegativeCountIsUnavailable()
        {
            RecordingSink sink = ActiveSink();

            Report(string.Empty, null, null, null, null, -1);

            string record = sink.Text["early_failure_0001.json"];
            StringAssert.Contains("\"capture_session_id_present\": false", record);
            StringAssert.Contains("\"private_capture_session_id\": null", record);
            StringAssert.Contains("\"private_declared_target\": null", record);
            StringAssert.Contains("\"stage\": null", record);
            StringAssert.Contains("\"failure_code\": null", record);
            StringAssert.Contains("\"exception_type\": null", record);
            StringAssert.Contains("\"buffered_observation_count\": null", record);
            StringAssert.Contains("\"buffered_observation_count_available\": false", record);
            StringAssert.DoesNotContain("\"stage\": \"\"", record);
            StringAssert.DoesNotContain("\"failure_code\": \"\"", record);
        }

        [Test]
        public void OverlongFieldsAreBounded()
        {
            RecordingSink sink = ActiveSink();

            Report(new string('x', 4000), "GREETING", "Capturing", "BridgeFailure", null, 1);

            string record = sink.Text["early_failure_0001.json"];
            Assert.Less(record.Length, 4000, "a caller must not be able to grow a record");
            StringAssert.Contains("~", record);
        }

        // ---- bounded history ------------------------------------------------------

        [Test]
        public void OverflowKeysAreCountedWithoutGrowingTheStoredHistory()
        {
            RecordingSink sink = ActiveSink();
            const int attempts = 600;
            for (int index = 0; index < attempts; index++)
                Report("capture_" + index, "GREETING", "Capturing", "BridgeFailure",
                       "IOException", index);

            Assert.AreEqual(FullHandEarlyFailureObserver.MaxRecords,
                            FullHandEarlyFailureObserver.RecordCount);
            Assert.AreEqual(FullHandEarlyFailureObserver.MaxRecords,
                            FullHandEarlyFailureObserver.RememberedKeyCount,
                            "the history never exceeds the records actually consumed");
            Assert.AreEqual(attempts - FullHandEarlyFailureObserver.MaxRecords,
                            FullHandEarlyFailureObserver.SuppressedOverflowCount);
            Assert.AreEqual(0, FullHandEarlyFailureObserver.SuppressedDuplicateCount);
            Assert.AreEqual(FullHandEarlyFailureObserver.MaxRecords, sink.Text.Count);

            // the very first key is still remembered after all that traffic
            Report("capture_0", "GREETING", "Capturing", "BridgeFailure", "IOException", 0);
            Assert.AreEqual(1, FullHandEarlyFailureObserver.SuppressedDuplicateCount);
            Assert.AreEqual(FullHandEarlyFailureObserver.MaxRecords,
                            FullHandEarlyFailureObserver.RememberedKeyCount);
            Assert.AreEqual(attempts - FullHandEarlyFailureObserver.MaxRecords,
                            FullHandEarlyFailureObserver.SuppressedOverflowCount);
        }

        [Test]
        public void RepeatedIdenticalFailuresAreDeduplicatedButDistinctFactsAreKept()
        {
            RecordingSink sink = ActiveSink();

            Report("capture_1", "GREETING", "Capturing", "BridgeFailure", "IOException", 5);
            Report("capture_1", "GREETING", "Capturing", "BridgeFailure", "IOException", 5);
            Report("capture_1", "GREETING", "Processing", "RecognitionFailure", null, 0);

            Assert.AreEqual(2, FullHandEarlyFailureObserver.RecordCount);
            Assert.AreEqual(1, FullHandEarlyFailureObserver.SuppressedDuplicateCount);
            Assert.AreEqual(2, sink.Text.Count);
        }

        [Test]
        public void AnObservationAndAnEarlyFailureForOneCaptureCoexistWithoutFalseClaims()
        {
            RecordingSink sink = ActiveSink();

            // an F5-shaped observation of this capture already exists and DID score
            sink.WriteText("observation_0001.json",
                           "{\"record_kind\": \"observation\", \"query_usable\": true}");
            Report("capture_9", "GREETING", "Processing", "RecognitionFailure", null, 87);

            Assert.IsTrue(sink.Text.ContainsKey("observation_0001.json"));
            string record = sink.Text["early_failure_0001.json"];
            Assert.AreEqual(2, sink.Text.Count, "the two record kinds never collide");
            // the failure note must not contradict the observation that already exists
            StringAssert.Contains("\"observer_scored\": false", record);
            StringAssert.Contains("\"attempt_model_and_snapshot_state\": "
                                  + "\"unobserved_by_this_diagnostic\"", record);
            StringAssert.DoesNotContain("nothing was scored", record);
        }

        [Test]
        public void WriteAndLogFaultsAreContainedAndNeverRetried()
        {
            var sink = new RecordingSink { ThrowOnWrite = true };
            FullHandEarlyFailureObserver.SinkForTest = sink;
            FullHandEarlyFailureObserver.LogForTest = line =>
            { throw new InvalidOperationException("injected log failure"); };

            Assert.DoesNotThrow(delegate
            {
                Report("capture_x", "GREETING", "Capturing", "BridgeFailure", "IOException", 3);
            });
            Assert.AreEqual(0, FullHandEarlyFailureObserver.RecordCount);
            Assert.AreEqual(1, FullHandEarlyFailureObserver.WriteFailureCount);

            sink.ThrowOnWrite = false;
            Report("capture_y", "GREETING", "Capturing", "BridgeFailure", "IOException", 3);
            Assert.IsFalse(sink.Text.ContainsKey("early_failure_0001.json"),
                           "the consumed number is never reused");
            Assert.IsTrue(sink.Text.ContainsKey("early_failure_0002.json"));
        }

        [Test]
        public void FileSinkRefusesToOverwriteAnExistingRecord()
        {
            string temporary = Path.Combine(Path.GetTempPath(),
                "wis_f6r1_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(temporary);
            try
            {
                var sink = new FullHandFileObservationSink(temporary);
                FullHandEarlyFailureObserver.SinkForTest = sink;
                FullHandEarlyFailureObserver.LogForTest = line => { };

                Report("capture_disk", "GREETING", "Capturing", "BridgeFailure", null, 7);
                Assert.IsTrue(File.Exists(Path.Combine(temporary, "early_failure_0001.json")));
                Assert.Throws<IOException>(
                    delegate { sink.WriteText("early_failure_0001.json", "{}"); });
            }
            finally
            {
                try { Directory.Delete(temporary, true); } catch (Exception) { }
            }
        }

        // ---- the real controller seam ---------------------------------------------

        [Test]
        public void ACapturingSourceFailureIsRecordedWithItsRealMetadata()
        {
            RecordingSink sink = ActiveSink();
            var source = new FakeObservationSource();
            using (var controller = new PersonalLiveRecognitionController(
                       source, new FakeRuntimeFactory(new FakeRuntime())))
            {
                Assert.IsTrue(controller.Initialize(Paths()));
                Assert.IsTrue(controller.BeginCapture("capture_live_1"));
                source.Emit(MakeValidObservation(0.10));
                source.Emit(MakeValidObservation(0.20));
                source.Emit(MakeValidObservation(0.30));
                Assert.AreEqual(3, controller.BufferedRawFrameCount);

                // the REAL SourceFailed event route: OnSourceFailure -> Fail
                source.Fail(new IOException("injected bridge failure"));

                Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
                Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            }

            Assert.AreEqual(1, FullHandEarlyFailureObserver.RecordCount);
            string record = sink.Text["early_failure_0001.json"];
            StringAssert.Contains("\"stage\": \"Capturing\"", record);
            StringAssert.Contains("\"failure_code\": \"BridgeFailure\"", record);
            StringAssert.Contains("\"exception_type\": \"IOException\"", record);
            StringAssert.Contains("\"private_capture_session_id\": \"capture_live_1\"", record);
            StringAssert.Contains("\"capture_session_id_present\": true", record);
            StringAssert.Contains("\"buffered_observation_count\": 3", record);
            StringAssert.Contains("\"private_declared_target\": \"GREETING\"", record);
            // the failure MESSAGE is never persisted
            StringAssert.DoesNotContain("injected bridge failure", record);
        }

        [Test]
        public void TheControllerFailureIsUnchangedWhenTheObserverItselfFaults()
        {
            FullHandEarlyFailureObserver.SinkForTest = new RecordingSink { ThrowOnWrite = true };
            FullHandEarlyFailureObserver.LogForTest = line =>
            { throw new InvalidOperationException("injected"); };

            var source = new FakeObservationSource();
            using (var controller = new PersonalLiveRecognitionController(
                       source, new FakeRuntimeFactory(new FakeRuntime())))
            {
                Assert.IsTrue(controller.Initialize(Paths()));
                Assert.IsTrue(controller.BeginCapture("capture_live_2"));
                source.Emit(MakeValidObservation(0.10));

                Assert.DoesNotThrow(delegate
                {
                    source.Fail(new IOException("injected bridge failure"));
                });
                Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
                Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
                Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure,
                                controller.PrimaryFailure.Code);
                Assert.AreEqual(1, FullHandEarlyFailureObserver.WriteFailureCount);
            }
        }

        [Test]
        public void AnInvalidTransitionBeforeAnyCaptureIsRecordedWithAnAbsentCaptureId()
        {
            RecordingSink sink = ActiveSink();
            using (var controller = new PersonalLiveRecognitionController(
                       new FakeObservationSource(),
                       new FakeRuntimeFactory(new FakeRuntime())))
            {
                Assert.IsFalse(controller.Reset(), "reset is invalid in state Idle");
                Assert.AreEqual(PersonalLiveFailureCode.InvalidStateTransition,
                                controller.Failure.Code);
            }

            string record = sink.Text["early_failure_0001.json"];
            StringAssert.Contains("\"stage\": \"Idle\"", record);
            StringAssert.Contains("\"failure_code\": \"InvalidStateTransition\"", record);
            StringAssert.Contains("\"capture_session_id_present\": false", record);
            StringAssert.Contains("\"private_capture_session_id\": null", record);
            StringAssert.Contains("\"buffered_observation_count\": 0", record);
            StringAssert.Contains("\"exception_type\": null", record);
            StringAssert.DoesNotContain("reset is invalid", record);
        }

        // ---- minimal fakes, following the approved integration fixture pattern -----

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

            public void Emit(RawMediaPipeObservation observation)
            {
                if (observations != null) observations(observation);
            }

            public void Fail(Exception exception)
            {
                if (failures != null) failures(exception);
            }
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime { }

        sealed class FakeRuntime : IPersonalLiveRecognitionRuntime
        {
            public bool IsPersonalActive { get { return true; } }
            public string ActiveProfileId { get { return "synthetic_profile"; } }

            public PersonalLiveRoutingResult Recognize(string sessionId,
                                                       RecognitionSequence sequence,
                                                       string expectedWord)
            {
                return SyntheticResult();
            }
        }

        sealed class FakeRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            readonly FakeRuntime runtime;

            public FakeRuntimeFactory(FakeRuntime runtime) { this.runtime = runtime; }

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
            {
                return new FakeGlobalRuntime();
            }

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths)
            {
                return runtime;
            }
        }

        static PersonalLiveRuntimePaths Paths()
        {
            // the fake factory ignores these; they only have to be non-empty
            return new PersonalLiveRuntimePaths("synthetic_templates",
                                                "synthetic_thresholds.json",
                                                "synthetic_profiles");
        }

        static PersonalLiveRoutingResult SyntheticResult()
        {
            var scores = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                { "GREETING", 0.12 }, { "LOVE", 0.34 }, { "THANK_YOU", 0.56 },
            };
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal)
            {
                { "GREETING", true }, { "LOVE", false }, { "THANK_YOU", false },
            };
            return new PersonalLiveRoutingResult("GREETING", true, true, true, scores,
                                                 accepts, 0.12, "prototype-GREETING",
                                                 new string[0]);
        }

        static double[] NaNCoordinates(int landmarkCount)
        {
            var values = new double[landmarkCount * 3];
            for (int index = 0; index < values.Length; index++) values[index] = double.NaN;
            return values;
        }

        static void WriteValid(double[] coordinates, bool[] valid, int landmark,
                               double x, double y, double z)
        {
            coordinates[landmark * 3] = x;
            coordinates[landmark * 3 + 1] = y;
            coordinates[landmark * 3 + 2] = z;
            valid[landmark] = true;
        }

        static RawMediaPipeObservation MakeValidObservation(double timestamp)
        {
            var poseCoordinates = NaNCoordinates(33);
            var poseValid = new bool[33];
            WriteValid(poseCoordinates, poseValid, 11, -1.0, 0.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 12, 1.0, 0.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 23, -1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 24, 1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 13, -1.2, -0.5, 0.1);
            WriteValid(poseCoordinates, poseValid, 14, 1.2, -0.5, 0.1);
            WriteValid(poseCoordinates, poseValid, 15, -1.4, -1.0, 0.2);
            WriteValid(poseCoordinates, poseValid, 16, 1.4, -1.0, 0.2);
            var handCoordinates = NaNCoordinates(21);
            var handValid = new bool[21];
            foreach (int landmark in new[] { 0, 4, 8, 12, 16, 20 })
                WriteValid(handCoordinates, handValid, landmark,
                           -1.0 + landmark * 0.02, -0.8, landmark * 0.005);
            return new RawMediaPipeObservation(
                timestamp, RawObservationConvention.LiveMediaPipe,
                new RawLandmarkObservationSet(33, poseCoordinates, poseValid,
                                              new bool[33], new bool[33]),
                new RawLandmarkObservationSet(21, handCoordinates, handValid,
                                              new bool[21], new bool[21]),
                RawLandmarkObservationSet.Absent(21));
        }
    }
}
