// F5R1 EditMode edge tests: no eviction, fully qualified paths, failure records and the
// loaded bundle identity. Synthetic geometry and the public synthetic bundle only.
// The original F5 tests stay unchanged and still apply. Nothing here has been executed.
using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandAttemptObserverR1Tests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;
        const string FixtureRelative = "Validation/m3_full_hand_20260909/f4_synthetic";

        // ---- synthetic geometry, same shape as the F5 fixtures ----------------

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

        static RawLandmarkObservationSet HandSet(int frame, double baseX, double baseY)
        {
            var c = NaNArray(Hand);
            var v = new bool[Hand];
            for (int joint = 0; joint < Hand; joint++)
                Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame,
                                 baseY + 0.003 * joint - 0.0005 * frame);
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], new bool[Hand]);
        }

        static List<RawMediaPipeObservation> Capture(int frames, bool geometry = true)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame), HandSet(frame, 0.30, 0.50), HandSet(frame, 0.60, 0.50),
                    geometry ? Standard() : null));
            return list;
        }

        sealed class RecordingSink : IFullHandObservationSink
        {
            public readonly Dictionary<string, byte[]> Bytes =
                new Dictionary<string, byte[]>(StringComparer.Ordinal);
            public readonly Dictionary<string, string> Text =
                new Dictionary<string, string>(StringComparer.Ordinal);
            public string ThrowForName;

            public void WriteBytes(string name, byte[] payload)
            {
                if (name == ThrowForName) throw new IOException("injected");
                Bytes[name] = (byte[])payload.Clone();
            }

            public void WriteText(string name, string text)
            {
                if (name == ThrowForName) throw new IOException("injected");
                Text[name] = text;
            }
        }

        static FullHandRuntimeBundle LoadFixtureBundle()
        {
            string project = Path.GetDirectoryName(Application.dataPath);
            string manifest = Path.Combine(
                Path.Combine(project, FixtureRelative.Replace('/', Path.DirectorySeparatorChar)),
                "runtime_manifest.json");
            if (!File.Exists(manifest))
                Assert.Ignore("the public synthetic bundle fixture was not found at "
                              + FixtureRelative);
            return FullHandRuntimeBundle.Load(manifest);
        }

        static FullHandObservationSession Session(RecordingSink sink)
        {
            return new FullHandObservationSession(LoadFixtureBundle(), sink, line => { });
        }

        [SetUp]
        public void SetUp() { FullHandAttemptObserver.ResetForTest(); }

        [TearDown]
        public void TearDown() { FullHandAttemptObserver.ResetForTest(); }

        // ---- 1. remembered ids are never evicted -------------------------------

        [Test]
        public void RememberedSessionIdsAreNeverEvictedOrReadmitted()
        {
            var sink = new RecordingSink();
            FullHandObservationSession session = Session(sink);
            var empty = new List<RawMediaPipeObservation>();
            const int distinct = 300;                 // comfortably past the old 256 bound
            string status;
            for (int index = 0; index < distinct; index++)
                session.TryObserve("session_" + index, "ANY_TARGET", empty, out status);

            Assert.AreEqual(distinct, session.ObservationCount);
            Assert.AreEqual(distinct, session.FailedCount);
            Assert.AreEqual(0, session.ScoredCount);
            Assert.AreEqual(distinct, sink.Text.Count);
            Assert.AreEqual(0, sink.Bytes.Count);

            Assert.IsFalse(session.TryObserve("session_0", "ANY_TARGET", empty, out status));
            Assert.AreEqual("ALREADY_OBSERVED", status);
            Assert.AreEqual(distinct, session.ObservationCount, "no new record number");
            Assert.AreEqual(distinct, sink.Text.Count, "no new record");
            Assert.AreEqual(1, session.SkippedDuplicateCount);
        }

        // ---- 2. fully qualified paths ------------------------------------------

        [Test]
        public void OnlyFullyQualifiedPathsAreAccepted()
        {
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified(null));
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified(string.Empty));
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified("relative.json"));
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified("sub/dir/file.json"));
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified("C:relative"),
                           "a drive-relative path resolves against that drive's directory");
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified("\\rootrelative"),
                           "a root-relative path resolves against the current drive");
            Assert.IsFalse(FullHandFileObservationSink.IsFullyQualified("file://x/y"));
            Assert.IsTrue(FullHandFileObservationSink.IsFullyQualified(Path.GetTempPath()));

            // refused before any normalization, whatever the current directory holds
            Assert.Throws<FullHandFormatException>(
                delegate { new FullHandFileObservationSink("relative_root"); });
            Assert.Throws<FullHandFormatException>(
                delegate { new FullHandFileObservationSink("C:relative_root"); });
        }

        [Test]
        public void ConfiguredPathsMustBeFullyQualified()
        {
            FullHandAttemptObserver.LogForTest = line => { };
            FullHandAttemptObserver.EnvironmentForTest = name =>
            {
                if (name == FullHandAttemptObserver.EnableVariable) return "1";
                if (name == FullHandAttemptObserver.ManifestVariable) return "C:relative.json";
                if (name == FullHandAttemptObserver.OutputVariable) return Path.GetTempPath();
                return null;
            };

            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("drive_relative", "ANY_TARGET", Capture(4));
            });
            Assert.AreEqual(FullHandObserverState.Faulted, FullHandAttemptObserver.State);
            Assert.AreEqual("BUNDLE_MANIFEST_NOT_ABSOLUTE", FullHandAttemptObserver.ReasonCode);
            Assert.AreEqual(0, FullHandAttemptObserver.LoadAttempts);
        }

        [Test]
        public void OnlyPlainRecordNamesAreWritten()
        {
            string temporary = Path.Combine(Path.GetTempPath(),
                "wis_f5r1_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(temporary);
            try
            {
                var sink = new FullHandFileObservationSink(temporary);
                foreach (string name in new[] { "sub/observation.json", "sub\\observation.json",
                                                "C:observation.json", "..", "." })
                    Assert.Throws<FullHandFormatException>(
                        delegate { sink.WriteText(name, "{}"); }, name);
            }
            finally
            {
                try { Directory.Delete(temporary, true); } catch (Exception) { }
            }
        }

        // ---- 3. failure records --------------------------------------------------

        [Test]
        public void EmptyAndRefusedCapturesWriteAMinimalFailureRecord()
        {
            var sink = new RecordingSink();
            FullHandObservationSession session = Session(sink);
            string status;

            session.TryObserve("empty_one", "TARGET_ALPHA", new List<RawMediaPipeObservation>(),
                               out status);
            Assert.AreEqual("EMPTY_CAPTURE", status);
            string first = sink.Text["observation_0001.json"];
            StringAssert.Contains("\"record_kind\": \"failure\"", first);
            StringAssert.Contains("\"status\": \"EMPTY_CAPTURE\"", first);
            StringAssert.Contains("\"scored\": false", first);
            StringAssert.Contains("\"binary_file\": null", first);
            StringAssert.Contains("\"private_declared_target\": \"TARGET_ALPHA\"", first);
            StringAssert.DoesNotContain("\"methods\"", first);
            StringAssert.DoesNotContain("binary_sha256", first);

            // no merged inference geometry: the adapter refuses and says why
            session.TryObserve("refused_one", "TARGET_OMEGA", Capture(5, false), out status);
            Assert.AreEqual("ADAPT_REFUSED", status);
            string second = sink.Text["observation_0002.json"];
            StringAssert.Contains("\"status\": \"ADAPT_REFUSED\"", second);
            StringAssert.Contains("geometry", second, "the actual adapter reason is retained");
            StringAssert.Contains("\"source_observation_count\": 5", second);
            StringAssert.Contains("\"frame_count\": -1", second);
            Assert.AreEqual(0, sink.Bytes.Count, "a failed input claims no binary");
        }

        [Test]
        public void FailedInputsAreCountedSeparatelyFromScoredOnes()
        {
            var sink = new RecordingSink();
            FullHandObservationSession session = Session(sink);
            string status;

            session.TryObserve("scored_one", "ANY_TARGET", Capture(8), out status);
            Assert.AreEqual("OBSERVED", status);
            session.TryObserve("empty_two", "ANY_TARGET", new List<RawMediaPipeObservation>(),
                               out status);

            Assert.AreEqual(2, session.ObservationCount);
            Assert.AreEqual(1, session.ScoredCount);
            Assert.AreEqual(1, session.FailedCount);
            Assert.AreEqual(1, sink.Bytes.Count, "only the scored input has a binary");
            Assert.AreEqual(2, sink.Text.Count);
        }

        [Test]
        public void AnExceptionRecordCarriesOnlyTheTypeAndPreservesTheFirstRecord()
        {
            var sink = new RecordingSink { ThrowForName = "observation_0001.json" };
            FullHandObservationSession session = Session(sink);

            string status = null;
            Assert.DoesNotThrow(delegate
            {
                session.TryObserve("io_fault", "ANY_TARGET", Capture(7), out status);
            });
            StringAssert.StartsWith("OBSERVER_ERROR_", status);
            Assert.AreEqual(1, session.FailedCount);
            Assert.AreEqual(0, session.ScoredCount);
            Assert.IsTrue(sink.Bytes.ContainsKey("observation_0001.bin"),
                          "the already written binary is preserved, never retried over");
            Assert.IsFalse(sink.Text.ContainsKey("observation_0001.json"));

            string record = sink.Text["observation_0001_error.json"];
            StringAssert.Contains("\"record_kind\": \"failure\"", record);
            StringAssert.Contains("IOException", record);
            StringAssert.DoesNotContain("injected", record,
                                        "only the exception TYPE may be recorded");
        }

        // ---- 4. loaded bundle identity -------------------------------------------

        [Test]
        public void RecordsCarryTheIdentityOfTheLoadedBundle()
        {
            var sink = new RecordingSink();
            FullHandRuntimeBundle bundle = LoadFixtureBundle();
            var session = new FullHandObservationSession(bundle, sink, line => { });

            string status;
            session.TryObserve("identity_one", "ANY_TARGET", Capture(8), out status);
            session.TryObserve("identity_two", "ANY_TARGET",
                               new List<RawMediaPipeObservation>(), out status);

            foreach (string name in new[] { "observation_0001.json", "observation_0002.json" })
            {
                string record = sink.Text[name];
                StringAssert.Contains("\"payload_sha256\": \"" + bundle.PayloadSha256 + "\"",
                                      record);
                StringAssert.Contains("\"synthetic\": true", record);
                StringAssert.Contains("\"reference_count\": " + bundle.ReferenceCount, record);
                StringAssert.Contains("\"vocabulary_count\": " + bundle.VocabularyCount, record);
                StringAssert.Contains("\"pair_schema\": \""
                                      + FullHandPairConstants.PairSchemaId + "\"", record);
                StringAssert.Contains("\"feature_point_count\": 86", record);
                StringAssert.Contains("\"resample_frames\": 32", record);
            }
        }
    }
}
