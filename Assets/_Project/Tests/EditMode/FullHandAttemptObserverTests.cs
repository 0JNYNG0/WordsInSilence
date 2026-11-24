// F5 EditMode tests for the opt-in full-hand observer.
//
// Synthetic geometry only, plus the public synthetic bundle fixture. No webcam, no real
// capture, no private model, no network, no scene. Writes go to an isolated temp directory
// or an in-memory sink. Nothing here has been executed.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandAttemptObserverTests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;
        const string FixtureRelative = "Validation/m3_full_hand_20260909/f4_synthetic";

        // ---- synthetic raw geometry, mirroring the F2 capture fixtures --------

        sealed class Geo
        {
            public int W = Width, H = Height, TexW = Width, TexH = Height,
                       CamW = Width, CamH = Height;
            public string Format = "RGBA32";
            public string RowOrder = "TopDown";
            public Step9DPreInferenceTransformState HFlip = Step9DPreInferenceTransformState.None;
            public Step9DPreInferenceTransformState VFlip = Step9DPreInferenceTransformState.None;
            public Step9DPreInferenceTransformState Crop = Step9DPreInferenceTransformState.None;
            public Step9DPreInferenceTransformState Resize = Step9DPreInferenceTransformState.None;
            public Step9DPreInferenceTransformState Letterbox = Step9DPreInferenceTransformState.None;
            public int? Rotation = 0;
            public int? PreRotation = 0;
            public bool? Mirrored = false;

            public Step9DInferenceFrameGeometry Build()
            {
                return new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.TestInjected,
                    CamW, CamH, 30, CamW, CamH, Rotation, Mirrored, TexW, TexH, W, H, Format,
                    HFlip, VFlip, PreRotation, Crop, Resize, Letterbox, 0L, 0L, 0, RowOrder);
            }
        }

        static Step9DMergedInferenceGeometry Standard()
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(new Geo().Build()),
                Step9DGeometryAcquisitionResult.ForAvailable(new Geo().Build()));
        }

        static double[] NaNArray(int count)
        {
            var values = new double[count * 3];
            for (int index = 0; index < values.Length; index++) values[index] = double.NaN;
            return values;
        }

        static void Put(double[] c, bool[] v, int landmark, double x, double y)
        {
            c[landmark * 3] = x; c[landmark * 3 + 1] = y; c[landmark * 3 + 2] = 0.0;
            v[landmark] = true;
        }

        static RawLandmarkObservationSet PoseSet(int frame)
        {
            var c = NaNArray(Pose);
            var v = new bool[Pose];
            double drift = 0.002 * frame;
            Put(c, v, 11, 0.35 + drift, 0.35);
            Put(c, v, 12, 0.65 + drift, 0.35);
            Put(c, v, 23, 0.40 + drift, 0.68);
            Put(c, v, 24, 0.60 + drift, 0.68);
            Put(c, v, 13, 0.30 + drift, 0.45);
            Put(c, v, 14, 0.70 + drift, 0.45);
            Put(c, v, 15, 0.28 + drift, 0.55);
            Put(c, v, 16, 0.72 + drift, 0.55);
            return new RawLandmarkObservationSet(Pose, c, v, new bool[Pose], new bool[Pose]);
        }

        static RawLandmarkObservationSet HandSet(int frame, double baseX, double baseY,
                                                 bool present = true)
        {
            var c = NaNArray(Hand);
            var v = new bool[Hand];
            if (present)
                for (int joint = 0; joint < Hand; joint++)
                    Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame,
                                     baseY + 0.003 * joint - 0.0005 * frame);
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], new bool[Hand]);
        }

        static List<RawMediaPipeObservation> Capture(int frames, bool hands = true)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame), HandSet(frame, 0.30, 0.50, hands),
                    HandSet(frame, 0.60, 0.50, hands), Standard()));
            return list;
        }

        // ---- sinks and bundle -------------------------------------------------

        sealed class RecordingSink : IFullHandObservationSink
        {
            public readonly Dictionary<string, byte[]> Bytes =
                new Dictionary<string, byte[]>(StringComparer.Ordinal);
            public readonly Dictionary<string, string> Text =
                new Dictionary<string, string>(StringComparer.Ordinal);
            public bool ThrowOnWrite;

            public void WriteBytes(string name, byte[] payload)
            {
                if (ThrowOnWrite) throw new IOException("injected write failure");
                Bytes[name] = (byte[])payload.Clone();
            }

            public void WriteText(string name, string text)
            {
                if (ThrowOnWrite) throw new IOException("injected write failure");
                Text[name] = text;
            }
        }

        static FullHandRuntimeBundle LoadFixtureBundle()
        {
            string project = Path.GetDirectoryName(Application.dataPath);
            string manifest = Path.Combine(project,
                FixtureRelative.Replace('/', Path.DirectorySeparatorChar));
            manifest = Path.Combine(manifest, "runtime_manifest.json");
            if (!File.Exists(manifest))
                Assert.Ignore("the public synthetic bundle fixture was not found at "
                              + FixtureRelative);
            return FullHandRuntimeBundle.Load(manifest);
        }

        [SetUp]
        public void SetUp() { FullHandAttemptObserver.ResetForTest(); }

        [TearDown]
        public void TearDown() { FullHandAttemptObserver.ResetForTest(); }

        // ---- configuration ----------------------------------------------------

        [Test]
        public void DisabledByDefaultLoadsNothingAndWritesNothing()
        {
            var sink = new RecordingSink();
            FullHandAttemptObserver.EnvironmentForTest = name => null;
            FullHandAttemptObserver.LogForTest = line =>
                Assert.Fail("a disabled observer must not log: " + line);

            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("session_a", "ANY_TARGET", Capture(6));
            });

            Assert.AreEqual(FullHandObserverState.Disabled, FullHandAttemptObserver.State);
            Assert.AreEqual("NOT_ENABLED", FullHandAttemptObserver.ReasonCode);
            Assert.AreEqual(0, FullHandAttemptObserver.LoadAttempts);
            Assert.AreEqual(0, sink.Bytes.Count);
            Assert.AreEqual(0, sink.Text.Count);
        }

        [Test]
        public void MissingOrRelativeConfigurationFaultsWithoutAffectingTheCaller()
        {
            var lines = new List<string>();
            FullHandAttemptObserver.LogForTest = lines.Add;
            FullHandAttemptObserver.EnvironmentForTest = name =>
                name == FullHandAttemptObserver.EnableVariable ? "1" : null;

            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("session_b", "ANY_TARGET", Capture(6));
            });
            Assert.AreEqual(FullHandObserverState.Faulted, FullHandAttemptObserver.State);
            Assert.AreEqual("BUNDLE_MANIFEST_NOT_ABSOLUTE", FullHandAttemptObserver.ReasonCode);
            Assert.AreEqual(0, FullHandAttemptObserver.LoadAttempts);
            Assert.AreEqual(1, lines.Count, "one short setup line only");
            StringAssert.DoesNotContain("session_b", lines[0]);

            FullHandAttemptObserver.ResetForTest();
            FullHandAttemptObserver.EnvironmentForTest = name =>
            {
                if (name == FullHandAttemptObserver.EnableVariable) return "1";
                if (name == FullHandAttemptObserver.ManifestVariable) return "relative.json";
                return null;
            };
            FullHandAttemptObserver.SafeObserve("session_c", "ANY_TARGET", Capture(6));
            Assert.AreEqual("BUNDLE_MANIFEST_NOT_ABSOLUTE", FullHandAttemptObserver.ReasonCode);
        }

        [Test]
        public void UnreadableEnvironmentIsContained()
        {
            FullHandAttemptObserver.LogForTest = line => { };
            FullHandAttemptObserver.EnvironmentForTest = name =>
            { throw new InvalidOperationException("injected"); };

            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("session_d", "ANY_TARGET", Capture(5));
            });
            Assert.AreEqual(FullHandObserverState.Faulted, FullHandAttemptObserver.State);
            Assert.AreEqual("ENVIRONMENT_UNREADABLE", FullHandAttemptObserver.ReasonCode);
        }

        [Test]
        public void OutputRootsUnderAssetsOrMissingAreRefused()
        {
            string temporary = Path.Combine(Path.GetTempPath(),
                "wis_f5_" + Guid.NewGuid().ToString("N"));
            string assets = Path.Combine(temporary, "Assets", "observations");
            Directory.CreateDirectory(assets);
            try
            {
                Assert.Throws<FullHandFormatException>(
                    delegate { new FullHandFileObservationSink(assets); });
                Assert.Throws<FullHandFormatException>(
                    delegate { new FullHandFileObservationSink(Path.Combine(temporary, "absent")); });
                var sink = new FullHandFileObservationSink(temporary);
                sink.WriteText("observation_0001.json", "{}");
                Assert.Throws<IOException>(
                    delegate { sink.WriteText("observation_0001.json", "{}"); },
                    "an existing file must never be overwritten");
            }
            finally
            {
                try { Directory.Delete(temporary, true); } catch (Exception) { }
            }
        }

        // ---- observation behaviour --------------------------------------------

        [Test]
        public void ObservesExactlyOncePerCaptureSessionId()
        {
            var sink = new RecordingSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });
            List<RawMediaPipeObservation> capture = Capture(8);

            string first, second;
            Assert.IsTrue(session.TryObserve("same_id", "ANY_TARGET", capture, out first), first);
            Assert.IsFalse(session.TryObserve("same_id", "ANY_TARGET", capture, out second));
            Assert.AreEqual("OBSERVED", first);
            Assert.AreEqual("ALREADY_OBSERVED", second);
            Assert.AreEqual(1, session.ObservationCount);
            Assert.AreEqual(1, session.SkippedDuplicateCount);
            Assert.AreEqual(1, sink.Bytes.Count);
            Assert.AreEqual(1, sink.Text.Count);
            Assert.IsTrue(sink.Text.ContainsKey("observation_0001.json"));
            Assert.IsTrue(sink.Bytes.ContainsKey("observation_0001.bin"));
        }

        [Test]
        public void DeclaredTargetIsAuditMetadataAndNeverChangesPredictions()
        {
            var sink = new RecordingSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });
            List<RawMediaPipeObservation> capture = Capture(9);

            string status;
            session.TryObserve("id_one", "TARGET_ALPHA", capture, out status);
            session.TryObserve("id_two", "TARGET_OMEGA", capture, out status);

            string first = sink.Text["observation_0001.json"];
            string second = sink.Text["observation_0002.json"];
            Assert.AreEqual(MethodsBlock(first), MethodsBlock(second),
                            "a different declared target must not change any prediction");
            StringAssert.Contains("\"private_declared_target\": \"TARGET_ALPHA\"", first);
            StringAssert.Contains("\"private_declared_target\": \"TARGET_OMEGA\"", second);
            StringAssert.Contains("audit metadata only", first);
        }

        [Test]
        public void UnusableSequenceIsReportedAsPerMethodTechnicalFailure()
        {
            var sink = new RecordingSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });

            string status;
            Assert.IsTrue(session.TryObserve("no_hands", "ANY_TARGET", Capture(7, false),
                                             out status), status);
            string json = sink.Text["observation_0001.json"];
            StringAssert.Contains("\"query_usable\": false", json);
            StringAssert.Contains("NO_LOCAL_HAND_SHAPE", json);
            foreach (string method in new[] { "dtw_only", "ml_only", "hybrid" })
                StringAssert.Contains("\"" + method + "\": {\"status\": \"TECHNICAL_FAILURE\"", json);
            StringAssert.DoesNotContain("\"status\": \"ACCEPT\"", json);
        }

        [Test]
        public void SourceObservationsAreNotMutated()
        {
            var sink = new RecordingSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });
            List<RawMediaPipeObservation> capture = Capture(6);
            var before = new List<double[]>();
            var beforeMasks = new List<bool[]>();
            foreach (RawMediaPipeObservation observation in capture)
            {
                before.Add(observation.Pose.CopyCoordinates());
                before.Add(observation.LeftHand.CopyCoordinates());
                before.Add(observation.RightHand.CopyCoordinates());
                beforeMasks.Add(observation.Pose.CopyValidMask());
                beforeMasks.Add(observation.LeftHand.CopyValidMask());
                beforeMasks.Add(observation.RightHand.CopyValidMask());
            }

            string status;
            session.TryObserve("immutable", "ANY_TARGET", capture, out status);

            int at = 0, maskAt = 0;
            foreach (RawMediaPipeObservation observation in capture)
            {
                Assert.AreEqual(before[at++], observation.Pose.CopyCoordinates());
                Assert.AreEqual(before[at++], observation.LeftHand.CopyCoordinates());
                Assert.AreEqual(before[at++], observation.RightHand.CopyCoordinates());
                Assert.AreEqual(beforeMasks[maskAt++], observation.Pose.CopyValidMask());
                Assert.AreEqual(beforeMasks[maskAt++], observation.LeftHand.CopyValidMask());
                Assert.AreEqual(beforeMasks[maskAt++], observation.RightHand.CopyValidMask());
            }
        }

        [Test]
        public void BinaryRecordRoundTripsExactlyAndMatchesTheRecordedDigest()
        {
            var sink = new RecordingSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });
            List<RawMediaPipeObservation> capture = Capture(11);

            string status;
            session.TryObserve("binary", "ANY_TARGET", capture, out status);
            byte[] binary = sink.Bytes["observation_0001.bin"];
            string json = sink.Text["observation_0001.json"];
            StringAssert.Contains("\"binary_sha256\": \"" +
                                  FullHandObservationSession.Sha256Hex(binary) + "\"", json);
            StringAssert.Contains("\"binary_bytes\": " + binary.Length, json);

            string failure;
            FullHandCommonCapture expected =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture, out failure);
            Assert.IsNotNull(expected, failure);

            int cursor = 0;
            foreach (char character in "WISFHOB1")
                Assert.AreEqual((byte)character, binary[cursor++]);
            Assert.AreEqual(1, ReadInt(binary, ref cursor));
            Assert.AreEqual(expected.FrameCount, ReadInt(binary, ref cursor));
            Assert.AreEqual(FullHandCommonSchema.PointCount, ReadInt(binary, ref cursor));
            Assert.AreEqual(FullHandCommonSchema.CoordinateDimensions, ReadInt(binary, ref cursor));
            for (int frame = 0; frame < expected.FrameCount; frame++)
                Assert.AreEqual(capture[frame].TimestampSeconds, ReadDouble(binary, ref cursor));

            double[] coordinates = expected.CopyCoordinates();
            for (int slot = 0; slot < coordinates.Length; slot++)
                Assert.AreEqual(BitConverter.DoubleToInt64Bits(coordinates[slot]),
                                BitConverter.DoubleToInt64Bits(ReadDouble(binary, ref cursor)),
                                "masked NaN and every valid coordinate must survive exactly");
            foreach (bool[] mask in new[] { expected.CopyValidMask(),
                                            expected.CopyOriginalValidMask(),
                                            expected.CopyInterpolatedMask() })
                for (int slot = 0; slot < mask.Length; slot++)
                    Assert.AreEqual(mask[slot], binary[cursor++] == 1);
            Assert.AreEqual(binary.Length, cursor, "the record has no trailing bytes");
        }

        [Test]
        public void InjectedSinkAndLogFaultsDoNotEscape()
        {
            var sink = new RecordingSink { ThrowOnWrite = true };
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink,
                line => { throw new InvalidOperationException("injected log failure"); });

            string status = null;
            Assert.DoesNotThrow(delegate
            {
                session.TryObserve("faulting", "ANY_TARGET", Capture(6), out status);
            });
            StringAssert.StartsWith("OBSERVER_ERROR_", status);

            FullHandAttemptObserver.SessionForTest = session;
            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("faulting_two", "ANY_TARGET", Capture(6));
            });
            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve(null, null, null);
            });
        }

        // ---- controller-shaped hook order -------------------------------------

        [Test]
        public void ObservationCannotChangeTheInputTheOldPathReceives()
        {
            List<RawMediaPipeObservation> capture = Capture(10);
            string baselineFailure;
            FullHandCommonCapture baseline =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture,
                                                                        out baselineFailure);
            Assert.IsNotNull(baseline, baselineFailure);

            // the hook order the controller uses: observe, then run the old preprocessing
            var sink = new RecordingSink { ThrowOnWrite = true };
            FullHandAttemptObserver.SessionForTest =
                new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });
            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("hook_order", "ANY_TARGET", capture);
            });

            string afterFailure;
            FullHandCommonCapture after =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture, out afterFailure);
            Assert.IsNotNull(after, afterFailure);
            Assert.AreEqual(baseline.FrameCount, after.FrameCount);
            Assert.AreEqual(baseline.ValidPointCount, after.ValidPointCount);
            double[] first = baseline.CopyCoordinates();
            double[] second = after.CopyCoordinates();
            for (int slot = 0; slot < first.Length; slot++)
                Assert.AreEqual(BitConverter.DoubleToInt64Bits(first[slot]),
                                BitConverter.DoubleToInt64Bits(second[slot]));
            Assert.AreEqual(baseline.CopyValidMask(), after.CopyValidMask());
        }

        // ---- helpers -----------------------------------------------------------

        static string MethodsBlock(string json)
        {
            int start = json.IndexOf("\"methods\": {", StringComparison.Ordinal);
            int end = json.IndexOf("\"elapsed_milliseconds\"", StringComparison.Ordinal);
            Assert.Greater(start, 0, "the record must carry a methods block");
            Assert.Greater(end, start);
            return json.Substring(start, end - start);
        }

        static int ReadInt(byte[] payload, ref int cursor)
        {
            int value = payload[cursor] | (payload[cursor + 1] << 8)
                      | (payload[cursor + 2] << 16) | (payload[cursor + 3] << 24);
            cursor += 4;
            return value;
        }

        static double ReadDouble(byte[] payload, ref int cursor)
        {
            double value = BitConverter.ToDouble(payload, cursor);
            cursor += 8;
            return value;
        }
    }
}
