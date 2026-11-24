// F12 EditMode tests for the opt-in bridge from the finalized full-hand observation to the
// F11 review-only candidate archive.
//
// Synthetic geometry only, plus the SAME public synthetic bundle fixture the existing F5
// tests use. No webcam, no private model, no learner data, no network, no scene. Archive
// writes go to an in-memory sink or to a temp directory this file creates and deletes.
// The existing F5 / F5R1 tests are untouched and still apply.
//
// Nothing here has been executed.
using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Personalization;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandReviewArchiveObserverTests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;
        const string FixtureRelative = "Validation/m3_full_hand_20260909/f4_synthetic";

        // ---- synthetic geometry, the same shape as the existing F5 fixtures ----

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
                                                 bool present)
        {
            var c = NaNArray(Hand);
            var v = new bool[Hand];
            if (present)
                for (int joint = 0; joint < Hand; joint++)
                    Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame,
                                     baseY + 0.003 * joint - 0.0005 * frame);
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], new bool[Hand]);
        }

        static List<RawMediaPipeObservation> Capture(int frames, bool hands = true,
                                                     bool geometry = true)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame), HandSet(frame, 0.30, 0.50, hands),
                    HandSet(frame, 0.60, 0.50, hands),
                    geometry ? Standard() : null));
            return list;
        }

        // ---- sinks and bundle ---------------------------------------------------

        sealed class RecordingSink : IFullHandObservationSink
        {
            public readonly Dictionary<string, byte[]> Bytes =
                new Dictionary<string, byte[]>(StringComparer.Ordinal);
            public readonly Dictionary<string, string> Text =
                new Dictionary<string, string>(StringComparer.Ordinal);

            public void WriteBytes(string name, byte[] payload)
            {
                Bytes[name] = (byte[])payload.Clone();
            }

            public void WriteText(string name, string text) { Text[name] = text; }
        }

        sealed class CapturingArchiveSink : IFullHandReviewCandidateSink
        {
            public readonly List<FullHandReviewCandidate> Saved =
                new List<FullHandReviewCandidate>();
            public bool ThrowOnWrite;

            public void Write(FullHandReviewCandidate candidate)
            {
                if (ThrowOnWrite) throw new IOException("injected archive failure");
                Saved.Add(candidate);
            }
        }

        static string FixtureManifest()
        {
            string project = Path.GetDirectoryName(Application.dataPath);
            string manifest = Path.Combine(
                Path.Combine(project, FixtureRelative.Replace('/', Path.DirectorySeparatorChar)),
                "runtime_manifest.json");
            if (!File.Exists(manifest))
                Assert.Ignore("the public synthetic bundle fixture was not found at "
                              + FixtureRelative);
            return manifest;
        }

        static FullHandRuntimeBundle LoadFixtureBundle()
        {
            return FullHandRuntimeBundle.Load(FixtureManifest());
        }

        /// <summary>The frozen hybrid entry for a capture, computed independently here.</summary>
        static FullHandMethodResult FrozenHybrid(FullHandRuntimeBundle bundle,
                                                 List<RawMediaPipeObservation> capture)
        {
            string failure;
            FullHandCommonCapture common =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture, out failure);
            Assert.IsNotNull(common, failure);
            FullHandEvaluation evaluation = FullHandScorer.Evaluate(
                bundle, FullHandPairGeometry.BuildFeatures(common));
            return evaluation.Method(FullHandMethods.Hybrid);
        }

        static string TempRoot()
        {
            string root = Path.Combine(Path.GetTempPath(),
                                       "wis_f12_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            return root;
        }

        [SetUp]
        public void SetUp() { FullHandAttemptObserver.ResetForTest(); }

        [TearDown]
        public void TearDown() { FullHandAttemptObserver.ResetForTest(); }

        // ---- 1. the archived capture is the observed capture --------------------

        [Test]
        public void ArchivedRecordCarriesExactlyTheLegacyObservationBinary()
        {
            var sink = new RecordingSink();
            var archive = new CapturingArchiveSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink,
                                                         line => { }, archive);
            List<RawMediaPipeObservation> capture = Capture(9);

            string status;
            Assert.IsTrue(session.TryObserve("archive_one", "TARGET_ALPHA", capture, out status),
                          status);
            Assert.AreEqual(1, archive.Saved.Count);
            Assert.AreEqual(1, session.ArchiveSavedCount);
            Assert.AreEqual(0, session.ArchiveFailedCount);
            Assert.AreEqual("ARCHIVED", session.LastArchiveStatus);

            FullHandReviewCandidate candidate = archive.Saved[0];
            Assert.AreEqual("observation_0001", candidate.SampleId);
            Assert.AreEqual("TARGET_ALPHA", candidate.DeclaredTargetWord);
            Assert.AreEqual(capture.Count, candidate.FrameCount);
            CollectionAssert.AreEqual(sink.Bytes["observation_0001.bin"],
                                      candidate.ToObservationBinary(),
                                      "the archive must reuse the same 46-point capture and " +
                                      "the same timestamps, byte for byte");
            double[] timestamps = candidate.CopyTimestamps();
            for (int frame = 0; frame < capture.Count; frame++)
                Assert.AreEqual(capture[frame].TimestampSeconds, timestamps[frame]);
        }

        // ---- 2. word evidence, and no promotion when target == prediction -------

        [Test]
        public void WordEvidenceMatchesTheFrozenHybridAndNeverBecomesEligible()
        {
            FullHandRuntimeBundle bundle = LoadFixtureBundle();
            List<RawMediaPipeObservation> capture = Capture(10);
            FullHandMethodResult hybrid = FrozenHybrid(bundle, capture);
            Assert.IsNotNull(hybrid);
            Assert.AreEqual(FullHandStatus.Ranked, hybrid.Status,
                            "this fixture capture is expected to rank");

            // the declared target is deliberately made equal to the prediction
            var archive = new CapturingArchiveSink();
            var session = new FullHandObservationSession(bundle, new RecordingSink(),
                                                         line => { }, archive);
            string status;
            session.TryObserve("archive_two", hybrid.Top1, capture, out status);
            Assert.AreEqual("OBSERVED", status);

            FullHandReviewCandidate candidate = archive.Saved[0];
            FullHandWordEvidence word = candidate.WordEvidence;
            Assert.IsTrue(word.IsScored);
            Assert.AreEqual(hybrid.Top1, word.PredictedWord);
            Assert.AreEqual(hybrid.Score, word.WordScore);
            Assert.AreEqual(bundle.PayloadSha256, word.ModelIdentity);
            Assert.AreEqual(FullHandPairConstants.PairSchemaId, word.FeatureSchemaId);
            Assert.AreEqual(candidate.DeclaredTargetWord, word.PredictedWord,
                            "the fixture for this test only");
            Assert.AreEqual(FullHandReviewQualityState.Unassessed,
                            candidate.QualityEvidence.State,
                            "agreement with the declared target is never a quality verdict");
            Assert.IsFalse(candidate.IsEnrollmentEligible);
            Assert.IsTrue(candidate.IsReviewOnly);
        }

        // ---- 3. the archive changes nothing when it is off ----------------------

        [Test]
        public void DisabledArchiveLeavesTheExistingRecordsAndCountersUnchanged()
        {
            FullHandRuntimeBundle bundle = LoadFixtureBundle();
            List<RawMediaPipeObservation> capture = Capture(8);

            var plainSink = new RecordingSink();
            var plain = new FullHandObservationSession(bundle, plainSink, line => { });
            var archivedSink = new RecordingSink();
            var archive = new CapturingArchiveSink();
            var archived = new FullHandObservationSession(bundle, archivedSink,
                                                          line => { }, archive);

            string first, second;
            plain.TryObserve("plain", "TARGET_ALPHA", capture, out first);
            archived.TryObserve("archived", "TARGET_ALPHA", capture, out second);

            Assert.AreEqual("OBSERVED", first);
            Assert.AreEqual("OBSERVED", second);
            Assert.AreEqual(0, plain.ArchiveSavedCount);
            Assert.AreEqual(0, plain.ArchiveFailedCount);
            Assert.AreEqual(string.Empty, plain.LastArchiveStatus);
            Assert.AreEqual(plain.ScoredCount, archived.ScoredCount);
            Assert.AreEqual(plain.FailedCount, archived.FailedCount);
            CollectionAssert.AreEquivalent(plainSink.Text.Keys, archivedSink.Text.Keys);
            CollectionAssert.AreEqual(plainSink.Bytes["observation_0001.bin"],
                                      archivedSink.Bytes["observation_0001.bin"],
                                      "the archive must not change the observation binary");
        }

        // ---- 4. a throwing archive cannot touch the observation -----------------

        [Test]
        public void ThrowingArchiveSinkPreservesObservedAndIsCountedSeparately()
        {
            var sink = new RecordingSink();
            var archive = new CapturingArchiveSink { ThrowOnWrite = true };
            var session = new FullHandObservationSession(LoadFixtureBundle(), sink,
                                                         line => { }, archive);

            string status = null;
            Assert.DoesNotThrow(delegate
            {
                session.TryObserve("archive_throws", "TARGET_ALPHA", Capture(7), out status);
            });
            Assert.AreEqual("OBSERVED", status);
            Assert.AreEqual(1, session.ScoredCount);
            Assert.AreEqual(0, session.FailedCount);
            Assert.AreEqual(0, session.ArchiveSavedCount);
            Assert.AreEqual(1, session.ArchiveFailedCount);
            StringAssert.StartsWith("ARCHIVE_FAILED_", session.LastArchiveStatus);
            StringAssert.DoesNotContain("injected", session.LastArchiveStatus,
                                        "only the exception TYPE may be reported");
            Assert.IsTrue(sink.Bytes.ContainsKey("observation_0001.bin"));
            Assert.IsTrue(sink.Text.ContainsKey("observation_0001.json"));
        }

        // ---- 5. exactly once, like the observation itself -----------------------

        [Test]
        public void ADuplicateCaptureIsArchivedAtMostOnce()
        {
            var archive = new CapturingArchiveSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(),
                                                         new RecordingSink(), line => { },
                                                         archive);
            List<RawMediaPipeObservation> capture = Capture(6);

            string first, second;
            Assert.IsTrue(session.TryObserve("same_id", "TARGET_ALPHA", capture, out first),
                          first);
            Assert.IsFalse(session.TryObserve("same_id", "TARGET_ALPHA", capture, out second));
            Assert.AreEqual("ALREADY_OBSERVED", second);
            Assert.AreEqual(1, archive.Saved.Count);
            Assert.AreEqual(1, session.ArchiveSavedCount);
        }

        // ---- 6. an unranked evaluation stays unscored ---------------------------

        [Test]
        public void ATechnicallyUnrankedEvaluationIsArchivedUnscored()
        {
            var archive = new CapturingArchiveSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(),
                                                         new RecordingSink(), line => { },
                                                         archive);

            string status;
            Assert.IsTrue(session.TryObserve("no_hands", "TARGET_ALPHA", Capture(7, false),
                                             out status), status);
            Assert.AreEqual(1, archive.Saved.Count);
            FullHandReviewCandidate candidate = archive.Saved[0];
            Assert.IsFalse(candidate.WordEvidence.IsScored,
                           "a technical failure must never be turned into a word label");
            Assert.AreEqual(string.Empty, candidate.WordEvidence.PredictedWord);
            Assert.AreEqual(FullHandReviewQualityState.Unassessed,
                            candidate.QualityEvidence.State);
            Assert.IsFalse(candidate.IsEnrollmentEligible);
        }

        // ---- 7. refused inputs archive nothing ----------------------------------

        [Test]
        public void EmptyRefusedAndUnlabelledCapturesProduceNoArchivedRecord()
        {
            var archive = new CapturingArchiveSink();
            var session = new FullHandObservationSession(LoadFixtureBundle(),
                                                         new RecordingSink(), line => { },
                                                         archive);
            string status;

            session.TryObserve("empty", "TARGET_ALPHA",
                               new List<RawMediaPipeObservation>(), out status);
            Assert.AreEqual("EMPTY_CAPTURE", status);
            session.TryObserve("refused", "TARGET_ALPHA", Capture(5, true, false), out status);
            Assert.AreEqual("ADAPT_REFUSED", status);
            Assert.AreEqual(0, archive.Saved.Count, "a failed input is never archived");
            Assert.AreEqual(0, session.ArchiveSavedCount);
            Assert.AreEqual(0, session.ArchiveFailedCount,
                            "the archive is not even attempted for a failed input");

            // an empty declared target is an archive failure, not a fabricated label
            session.TryObserve("no_target", string.Empty, Capture(6), out status);
            Assert.AreEqual("OBSERVED", status, "the observation itself is unaffected");
            Assert.AreEqual(0, archive.Saved.Count);
            Assert.AreEqual(1, session.ArchiveFailedCount);
            StringAssert.StartsWith("ARCHIVE_FAILED_", session.LastArchiveStatus);
        }

        // ---- 8. the file sink -----------------------------------------------------

        [Test]
        public void FileArchiveSinkRoundTripsUnderAnOwnedTempRootAndNeverOverwrites()
        {
            FullHandRuntimeBundle bundle = LoadFixtureBundle();
            string root = TempRoot();
            try
            {
                var archive = new FullHandFileReviewCandidateSink(root);
                var session = new FullHandObservationSession(bundle, new RecordingSink(),
                                                             line => { }, archive);
                List<RawMediaPipeObservation> capture = Capture(8);
                string status;
                Assert.IsTrue(session.TryObserve("file_one", "TARGET_ALPHA", capture,
                                                 out status), status);
                Assert.AreEqual(1, session.ArchiveSavedCount);

                string path = Path.Combine(root, "observation_0001" +
                                                 FullHandReviewCandidateStore.FileExtension);
                Assert.IsTrue(File.Exists(path));
                FullHandReviewCandidate loaded = FullHandReviewCandidateStore.Load(path);
                Assert.AreEqual("observation_0001", loaded.SampleId);
                Assert.AreEqual("TARGET_ALPHA", loaded.DeclaredTargetWord);
                Assert.AreEqual(capture.Count, loaded.FrameCount);
                Assert.IsFalse(loaded.IsEnrollmentEligible);

                // the same sample id is never replaced or retried over
                Assert.Throws<IOException>(delegate { archive.Write(loaded); });
                Assert.AreEqual(1, Directory.GetFiles(root).Length);
            }
            finally
            {
                try { Directory.Delete(root, true); } catch (Exception) { }
            }
        }

        [Test]
        public void FileArchiveSinkRefusesMissingRelativeAndAssetsRoots()
        {
            string root = TempRoot();
            string assets = Path.Combine(root, "Assets", "review");
            Directory.CreateDirectory(assets);
            try
            {
                Assert.Throws<FullHandFormatException>(
                    delegate { new FullHandFileReviewCandidateSink(assets); });
                Assert.Throws<FullHandFormatException>(
                    delegate { new FullHandFileReviewCandidateSink("relative_root"); });
                Assert.Throws<FullHandFormatException>(
                    delegate
                    {
                        new FullHandFileReviewCandidateSink(Path.Combine(root, "absent"));
                    });
                Assert.DoesNotThrow(delegate { new FullHandFileReviewCandidateSink(root); });
            }
            finally
            {
                try { Directory.Delete(root, true); } catch (Exception) { }
            }
        }

        // ---- 9. configuration ------------------------------------------------------

        [Test]
        public void ADisabledObserverNeverReadsAnArchiveVariable()
        {
            var read = new List<string>();
            FullHandAttemptObserver.LogForTest = line =>
                Assert.Fail("a disabled observer must not log: " + line);
            FullHandAttemptObserver.EnvironmentForTest = name => { read.Add(name); return null; };

            Assert.DoesNotThrow(delegate
            {
                FullHandAttemptObserver.SafeObserve("disabled", "TARGET_ALPHA", Capture(6));
            });
            Assert.AreEqual(FullHandObserverState.Disabled, FullHandAttemptObserver.State);
            CollectionAssert.DoesNotContain(read,
                FullHandAttemptObserver.ArchiveEnableVariable);
            CollectionAssert.DoesNotContain(read,
                FullHandAttemptObserver.ArchiveOutputVariable);
            Assert.IsFalse(FullHandAttemptObserver.ArchiveEnabled);
        }

        [Test]
        public void AnInvalidArchiveConfigurationDisablesOnlyTheArchive()
        {
            string manifest = FixtureManifest();
            string output = TempRoot();
            try
            {
                var lines = new List<string>();
                FullHandAttemptObserver.LogForTest = lines.Add;
                FullHandAttemptObserver.EnvironmentForTest = name =>
                {
                    if (name == FullHandAttemptObserver.EnableVariable) return "1";
                    if (name == FullHandAttemptObserver.ManifestVariable) return manifest;
                    if (name == FullHandAttemptObserver.OutputVariable) return output;
                    if (name == FullHandAttemptObserver.ArchiveEnableVariable) return "1";
                    if (name == FullHandAttemptObserver.ArchiveOutputVariable)
                        return "relative_archive_root";
                    return null;
                };

                Assert.DoesNotThrow(delegate
                {
                    FullHandAttemptObserver.SafeObserve("config", "TARGET_ALPHA", Capture(6));
                });
                Assert.AreEqual(FullHandObserverState.Ready, FullHandAttemptObserver.State,
                                "a bad archive root must not fault the observer");
                Assert.AreEqual(string.Empty, FullHandAttemptObserver.ReasonCode);
                Assert.IsFalse(FullHandAttemptObserver.ArchiveEnabled);
                Assert.AreEqual("ARCHIVE_ROOT_NOT_ABSOLUTE",
                                FullHandAttemptObserver.ArchiveReasonCode);
                Assert.IsTrue(Directory.GetFiles(output).Length > 0,
                              "the normal observation records are still written");
                foreach (string line in lines)
                    StringAssert.DoesNotContain("relative_archive_root", line,
                                                "a status line carries no path");
            }
            finally
            {
                try { Directory.Delete(output, true); } catch (Exception) { }
            }
        }
    }
}
