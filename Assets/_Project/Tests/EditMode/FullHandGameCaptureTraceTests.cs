// F22 EditMode tests for the local capture-evidence writer.
//
// Everything here is synthetic and in memory except ONE isolated temporary directory that
// each test creates and deletes itself. No camera, no persistentDataPath, no project file,
// no private learner data, no model file, no real profile, no fixture and no network. The
// bank, references and gates are constructed by the test, so no production cutoff and no
// fitted number is asserted anywhere.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandGameCaptureTraceTests
    {
        const int SourceFrames = 4;
        const string Participant = "participant_a";
        static readonly string[] Words = { "kiwi", "zebra" };   // arbitrary labels, no word rules

        // raw-capture helper constants (synthetic geometry only)
        const int Pose = 33;
        const int Hand = 21;
        const int Width = 1280;
        const int Height = 720;

        string root;

        [SetUp]
        public void CreateIsolatedDirectory()
        {
            root = Path.Combine(Path.GetTempPath(),
                                "wis_f22_trace_" + Guid.NewGuid().ToString("N"));
        }

        [TearDown]
        public void RemoveIsolatedDirectory()
        {
            try { if (Directory.Exists(root)) Directory.Delete(root, true); }
            catch (Exception) { /* a leftover temp directory is not a test failure */ }
        }

        // ---- synthetic pipeline material ---------------------------------------

        static void Arrays(double seed, out double[] coordinates, out bool[] mask)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            coordinates = new double[SourceFrames * points * dims];
            mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = seed + point * 0.13 + frame * 0.05;
                    coordinates[slot * dims + 1] = seed * 0.5 + point * 0.07 - frame * 0.03;
                    coordinates[slot * dims + 2] = 0.0;
                }
        }

        static FullHandFeatureSequence Sequence(double seed)
        {
            double[] coordinates; bool[] mask;
            Arrays(seed, out coordinates, out mask);
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, SourceFrames);
        }

        static FullHandCommonCapture Capture(double seed)
        {
            double[] coordinates; bool[] mask;
            Arrays(seed, out coordinates, out mask);
            // (frameCount, coordinates, validMask, originalValidMask, interpolatedMask,
            //  validPointCount, interpolatedPointCount, width, height, convention)
            return new FullHandCommonCapture(SourceFrames, coordinates, mask, mask,
                new bool[mask.Length], mask.Length, 0, 640, 480, "test_convention");
        }

        /// <summary>
        /// Nothing is valid: validPointCount and interpolatedPointCount are both 0. F20's
        /// BuildFeatures accepts this empty-validity sequence and the scorer reports missing
        /// evidence rather than a score.
        /// </summary>
        static FullHandCommonCapture UnusableCapture()
        {
            double[] coordinates; bool[] mask;
            Arrays(2.0, out coordinates, out mask);
            var none = new bool[mask.Length];
            return new FullHandCommonCapture(SourceFrames, coordinates, none, none,
                new bool[mask.Length], 0, 0, 640, 480, "test_convention");
        }

        static FullHandModel Model(int width)
        {
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            for (int index = 0; index < width; index++)
            {
                mean[index] = 0.1 * index;
                scale[index] = 1.0 + 0.01 * index;
                coefficients[index] = ((index % 3) - 1) * 0.25;
            }
            return new FullHandModel(mean, scale, coefficients, 0.2);
        }

        static FullHandInMemoryCommonBank Bank()
        {
            return new FullHandInMemoryCommonBank("model_test_v1", Words,
                new[] { "common_0", "common_1", "common_2", "common_3" },
                new[] { 0, 0, 1, 1 },
                new[] { Sequence(1.0), Sequence(1.4), Sequence(3.0), Sequence(3.4) },
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
        }

        static string Hash(int value) { return value.ToString("x").PadLeft(64, '0'); }

        static FullHandGateConfig GatesAt(string id, double quality)
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.AtQualityThreshold(id + "_" + index,
                    FullHandMethods.All[index], quality);
            return new FullHandGateConfig(id, gates);
        }

        static FullHandRecognitionPipeline Pipeline(params FullHandSample[] additions)
        {
            var set = additions == null || additions.Length == 0
                ? FullHandParticipantReferenceSet.Empty(Participant)
                : new FullHandParticipantReferenceSet(Participant, additions);
            return new FullHandRecognitionPipeline(Bank(), GatesAt("config_test", 0.5),
                                                   Participant, set);
        }

        static FullHandSample Addition(double seed, string id, int hash, string label)
        {
            return new FullHandSample(id, Participant, Hash(hash), label, Sequence(seed));
        }

        /// <summary>Stand-in capture bytes for the writer-only tests.</summary>
        static byte[] SyntheticBinary(byte seed)
        {
            var bytes = new byte[64];
            for (int index = 0; index < bytes.Length; index++)
                bytes[index] = (byte)(seed + index);
            return bytes;
        }

        static FullHandGameCaptureResult Outcome(string captureId, string declaredTarget,
                                                 byte[] binary, FullHandRecognitionPipeline pipeline,
                                                 FullHandCommonCapture capture)
        {
            string sourceSha = FullHandCaptureTraceWriter.Sha256Hex(binary);
            FullHandRecognitionOutcome outcome =
                pipeline.Evaluate(capture, captureId, sourceSha, declaredTarget);
            return FullHandGameCaptureResult.ForOutcome(captureId, SourceFrames, outcome,
                                                        sourceSha, binary);
        }

        /// <summary>A processor that returns exactly the result it was handed.</summary>
        sealed class FixedProcessor : IFullHandGameCaptureProcessor
        {
            readonly FullHandGameCaptureResult result;
            public int Calls { get; private set; }
            public FixedProcessor(FullHandGameCaptureResult result) { this.result = result; }
            public FullHandGameCaptureResult Process(string captureSessionId, string declaredTarget,
                IReadOnlyList<RawMediaPipeObservation> observations)
            {
                Calls++;
                return result;
            }
        }

        // ---- synthetic raw capture (from the approved diagnostics helper) -------

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

        static List<RawMediaPipeObservation> RawCapture(int frames)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame), HandSet(frame, 0.30, 0.50),
                    HandSet(frame, 0.60, 0.50), Standard()));
            return list;
        }

        // ---- file helpers ------------------------------------------------------

        string[] Manifests()
        {
            return Directory.Exists(root)
                ? Directory.GetFiles(root, "*" + FullHandCaptureTraceWriter.ManifestExtension)
                : new string[0];
        }

        string[] Binaries()
        {
            return Directory.Exists(root)
                ? Directory.GetFiles(root, "*" + FullHandCaptureTraceWriter.BinaryExtension)
                : new string[0];
        }

        string[] StagingFiles()
        {
            return Directory.Exists(root)
                ? Directory.GetFiles(root, "*" + FullHandCaptureTraceWriter.StagingExtension)
                : new string[0];
        }

        static string ReadText(string path)
        {
            return new UTF8Encoding(false).GetString(File.ReadAllBytes(path));
        }

        Dictionary<string, object> ParsedManifest()
        {
            Assert.AreEqual(1, Manifests().Length, "expected exactly one completed manifest");
            return (Dictionary<string, object>)TinyJson.Parse(ReadText(Manifests()[0]));
        }

        // ---- tests -------------------------------------------------------------

        [Test]
        public void BinaryAndManifestArePairedByExactByteCountAndHash()
        {
            byte[] binary = SyntheticBinary(7);
            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.None,
                writer.Record(Outcome("capture_1", "kiwi", binary, Pipeline(), Capture(2.0))));

            Assert.AreEqual(1, Binaries().Length);
            Assert.AreEqual(1, Manifests().Length);
            Assert.AreEqual(0, StagingFiles().Length, "no staging file may survive a success");

            byte[] written = File.ReadAllBytes(Binaries()[0]);
            CollectionAssert.AreEqual(binary, written, "the exact retained binary must be reused");

            Dictionary<string, object> manifest = ParsedManifest();
            Assert.AreEqual(FullHandCaptureTraceWriter.Schema, manifest["schema"]);
            var source = (Dictionary<string, object>)manifest["source_binary"];
            Assert.AreEqual((double)binary.Length, source["byte_count"]);
            Assert.AreEqual(FullHandCaptureTraceWriter.Sha256Hex(binary), source["sha256"]);
            Assert.AreEqual(FullHandCaptureTraceWriter.Sha256Hex(binary),
                            source["source_sha256_from_pipeline"]);
            Assert.AreEqual(Path.GetFileName(Binaries()[0]), source["file"]);

            // a requested expression is never promoted to a label or to enrollment
            Assert.AreEqual("UNASSESSED", manifest["quality_label"]);
            Assert.AreEqual(false, manifest["eligible_for_enrollment"]);
            Assert.AreEqual(false, manifest["declared_target_is_ground_truth"]);
            Assert.AreEqual(true, manifest["declared_target_is_requested_expression"]);
        }

        [Test]
        public void ManifestCarriesBothBanksAndAllThreeMethodsForTheSameQueryDigest()
        {
            var pipeline = Pipeline(Addition(1.2, "addition_0", 11, Words[0]));
            byte[] binary = SyntheticBinary(3);
            var writer = new FullHandCaptureTraceWriter(root);
            FullHandGameCaptureResult result =
                Outcome("capture_same_query", "kiwi", binary, pipeline, Capture(2.0));
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(result));

            Dictionary<string, object> manifest = ParsedManifest();
            var banks = (Dictionary<string, object>)manifest["banks"];
            var common = (Dictionary<string, object>)banks["common_only"];
            var withParticipant = (Dictionary<string, object>)banks["common_plus_current_participant"];

            string digest = result.Outcome.QueryFeatureDigest;
            Assert.AreEqual(digest, manifest["query_feature_digest"]);
            foreach (Dictionary<string, object> bank in new[] { common, withParticipant })
            {
                var methods = (List<object>)bank["methods"];
                Assert.AreEqual(FullHandMethods.All.Length, methods.Count);
                for (int index = 0; index < FullHandMethods.All.Length; index++)
                {
                    var entry = (Dictionary<string, object>)methods[index];
                    Assert.AreEqual(FullHandMethods.All[index], entry["method"]);
                    // one capture, one participant, ONE query digest for every method
                    Assert.AreEqual(digest, entry["query_feature_digest"]);
                    Assert.IsTrue(entry.ContainsKey("status"));
                    Assert.IsTrue(entry.ContainsKey("decision"));
                    Assert.IsTrue(entry.ContainsKey("reason"));
                    Assert.IsTrue(entry.ContainsKey("gate_id"));
                    Assert.IsTrue(entry.ContainsKey("top1"));
                    Assert.IsTrue(entry.ContainsKey("class_scores"));
                    Assert.IsTrue(entry.ContainsKey("classes_without_evidence"));
                }
                Assert.IsTrue(bank.ContainsKey("excluded_common_reference_count"));
                Assert.IsTrue(bank.ContainsKey("excluded_participant_reference_count"));
            }

            Assert.AreEqual(result.Outcome.CommonModelId, manifest["common_model_id"]);
            Assert.AreEqual(result.Outcome.GateConfigId, manifest["gate_config_id"]);
            Assert.AreEqual(Participant, manifest["participant_id"]);
            Assert.AreEqual((double)result.Outcome.ScoringPasses, manifest["scoring_passes"]);
            Assert.AreEqual(result.Outcome.ReusedCommonEvaluation,
                            manifest["reused_common_evaluation"]);
            // this bank really did carry a participant addition, so the claim is allowed
            Assert.AreEqual(true, manifest["personalization_claimed"]);
        }

        [Test]
        public void ClassScoresKeepTheBanksOwnOrderAndValuesWithoutReranking()
        {
            var pipeline = Pipeline();
            byte[] binary = SyntheticBinary(13);
            var writer = new FullHandCaptureTraceWriter(root);
            FullHandGameCaptureResult result =
                Outcome("capture_class_scores", "kiwi", binary, pipeline, Capture(2.0));
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(result));

            var banks = (Dictionary<string, object>)ParsedManifest()["banks"];
            var methods = (List<object>)((Dictionary<string, object>)banks["common_only"])["methods"];

            for (int index = 0; index < FullHandMethods.All.Length; index++)
            {
                FullHandMethodDecision decision =
                    result.Outcome.Before.Decision(FullHandMethods.All[index]);
                string[] classes = decision.CopyClasses();
                double[] raw = decision.CopyRawScores();
                var entry = (Dictionary<string, object>)methods[index];
                var written = (List<object>)entry["class_scores"];

                Assert.AreEqual(classes.Length, written.Count);
                for (int slot = 0; slot < classes.Length; slot++)
                {
                    var pair = (Dictionary<string, object>)written[slot];
                    // same class in the same slot: the association is never re-sorted
                    Assert.AreEqual(classes[slot], pair["class"]);
                    if (double.IsNaN(raw[slot]) || double.IsInfinity(raw[slot]))
                    {
                        Assert.IsNull(pair["score"]);
                        Assert.AreEqual(false, pair["score_finite"]);
                    }
                    else
                    {
                        Assert.AreEqual(raw[slot], (double)pair["score"], 1e-12);
                        Assert.AreEqual(true, pair["score_finite"]);
                    }
                }
                Assert.AreEqual(decision.Top1, entry["top1"]);
            }
        }

        [Test]
        public void NoPersonalizationIsClaimedWithoutParticipantReferences()
        {
            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(
                Outcome("capture_common_only", null, SyntheticBinary(1), Pipeline(), Capture(2.0))));

            Dictionary<string, object> manifest = ParsedManifest();
            Assert.AreEqual(false, manifest["personalization_claimed"]);
            var banks = (Dictionary<string, object>)manifest["banks"];
            var withParticipant =
                (Dictionary<string, object>)banks["common_plus_current_participant"];
            Assert.AreEqual(0.0, withParticipant["participant_reference_count"]);
            // an undeclared target stays absent instead of becoming an empty expression
            Assert.IsNull(manifest["declared_target"]);
        }

        [Test]
        public void ACaptureIdIsRecordedAtMostOncePerWriterSession()
        {
            var writer = new FullHandCaptureTraceWriter(root);
            var pipeline = Pipeline();
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(
                Outcome("capture_dup", "kiwi", SyntheticBinary(4), pipeline, Capture(2.0))));
            Assert.AreEqual(FullHandTraceCodes.DuplicateCapture, writer.Record(
                Outcome("capture_dup", "kiwi", SyntheticBinary(4), pipeline, Capture(2.0))));

            Assert.AreEqual(1, writer.RecordCount);
            Assert.AreEqual(1, Manifests().Length);
            Assert.AreEqual(1, Binaries().Length);
        }

        [Test]
        public void DistinctCapturesGetDistinctFilesAndNothingIsOverwritten()
        {
            var writer = new FullHandCaptureTraceWriter(root);
            var pipeline = Pipeline();
            writer.Record(Outcome("capture_a", "kiwi", SyntheticBinary(5), pipeline, Capture(2.0)));
            writer.Record(Outcome("capture_b", "zebra", SyntheticBinary(9), pipeline, Capture(2.2)));

            Assert.AreEqual(2, Manifests().Length);
            Assert.AreEqual(2, Binaries().Length);
            Assert.AreEqual(2, writer.RecordCount);
            Assert.AreEqual(0, StagingFiles().Length);
            CollectionAssert.AllItemsAreUnique(Binaries());
        }

        [Test]
        public void GeneratedNamesNeverContainTheParticipantOrTheDeclaredTarget()
        {
            var writer = new FullHandCaptureTraceWriter(root);
            writer.Record(Outcome("capture_naming", "kiwi", SyntheticBinary(6), Pipeline(),
                                  Capture(2.0)));
            string name = Path.GetFileName(Manifests()[0]);
            StringAssert.DoesNotContain(Participant, name);
            StringAssert.DoesNotContain("kiwi", name);
            StringAssert.DoesNotContain("capture_naming", name);
        }

        [Test]
        public void ATraceFailureLeavesTheRecognitionResultUntouched()
        {
            // a relative path is not a usable fully qualified output directory
            var writer = new FullHandCaptureTraceWriter("relative\\not\\qualified");
            FullHandGameCaptureResult original =
                Outcome("capture_fail", "kiwi", SyntheticBinary(8), Pipeline(), Capture(2.0));
            var inner = new FixedProcessor(original);
            var processor = new FullHandGameCaptureTracingProcessor(inner, writer);

            FullHandGameCaptureResult returned = processor.Process("capture_fail", "kiwi", null);

            Assert.AreSame(original, returned, "the wrapper must return the original result");
            Assert.IsTrue(returned.Succeeded);
            Assert.AreSame(original.Outcome, returned.Outcome);
            Assert.AreEqual(1, inner.Calls, "no hidden retry and no second scoring pass");
            Assert.AreEqual(FullHandTraceCodes.DirectoryUnusable, processor.LastWriteCode);
            Assert.AreEqual(0, processor.RecordCount);
            Assert.AreEqual(0, Manifests().Length);
            Assert.AreEqual(0, StagingFiles().Length);
        }

        [Test]
        public void AMismatchedBinaryHashProducesNoCompletedPair()
        {
            var pipeline = Pipeline();
            byte[] binary = SyntheticBinary(21);
            FullHandRecognitionOutcome outcome =
                pipeline.Evaluate(Capture(2.0), "capture_mismatch", Hash(77), "kiwi");
            // the retained bytes are NOT the bytes this hash describes
            FullHandGameCaptureResult result = FullHandGameCaptureResult.ForOutcome(
                "capture_mismatch", SourceFrames, outcome, Hash(77), binary);

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.BinaryHashMismatch, writer.Record(result));
            Assert.AreEqual(0, Manifests().Length);
            Assert.AreEqual(0, Binaries().Length);
            Assert.AreEqual(0, writer.RecordCount);
        }

        [Test]
        public void AnAbsentSourceHashProducesNoCompletedPair()
        {
            var pipeline = Pipeline();
            FullHandRecognitionOutcome outcome =
                pipeline.Evaluate(Capture(2.0), "capture_no_hash", Hash(3), "kiwi");
            FullHandGameCaptureResult result = FullHandGameCaptureResult.ForOutcome(
                "capture_no_hash", SourceFrames, outcome, null, SyntheticBinary(22));

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.BinaryHashMismatch, writer.Record(result));
            Assert.AreEqual(0, Manifests().Length);
        }

        [Test]
        public void ASuccessfulOutcomeWithoutARetainedBinaryIsNotWritten()
        {
            var pipeline = Pipeline();
            FullHandRecognitionOutcome outcome =
                pipeline.Evaluate(Capture(2.0), "capture_no_binary", Hash(5), "kiwi");
            // the legacy (retention-off) factory: no binary is kept after scoring
            FullHandGameCaptureResult result =
                FullHandGameCaptureResult.ForOutcome("capture_no_binary", SourceFrames, outcome);
            Assert.IsFalse(result.HasSourceBinary);
            Assert.IsNull(result.CopySourceBinary());

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.BinaryUnavailable, writer.Record(result));
            Assert.AreEqual(0, Manifests().Length, "no unpaired manifest may exist");
            Assert.AreEqual(0, Binaries().Length);
        }

        [Test]
        public void TheRetainedBinaryIsADefensiveImmutableCopy()
        {
            byte[] source = SyntheticBinary(2);
            var original = (byte[])source.Clone();
            FullHandGameCaptureResult result =
                Outcome("capture_copy", "kiwi", source, Pipeline(), Capture(2.0));

            // mutating the caller's array cannot reach the retained evidence
            for (int index = 0; index < source.Length; index++) source[index] = 0xFF;
            CollectionAssert.AreEqual(original, result.CopySourceBinary());

            // and the exposed copy is a fresh array each time
            byte[] first = result.CopySourceBinary();
            for (int index = 0; index < first.Length; index++) first[index] = 0x00;
            CollectionAssert.AreEqual(original, result.CopySourceBinary());

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(result));
            CollectionAssert.AreEqual(original, File.ReadAllBytes(Binaries()[0]));
        }

        [Test]
        public void ATechnicalRefusalIsPreservedWithoutScoresOrBinary()
        {
            FullHandGameCaptureResult refusal = FullHandGameCaptureResult.ForRefusal(
                "capture_refused", 0, FullHandBridgeCodes.EmptyCapture,
                "the capture carried no observation");

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(refusal));

            Assert.AreEqual(1, Manifests().Length);
            Assert.AreEqual(0, Binaries().Length, "a refusal has no capture binary");

            Dictionary<string, object> manifest = ParsedManifest();
            Assert.AreEqual("technical_refusal", manifest["record_kind"]);
            Assert.AreEqual(false, manifest["scores_present"]);
            Assert.IsNull(manifest["source_binary"]);
            Assert.AreEqual(false, manifest["personalization_claimed"]);
            var failure = (Dictionary<string, object>)manifest["failure"];
            Assert.AreEqual(FullHandBridgeCodes.EmptyCapture, failure["code"]);
            // nothing is invented for a refusal: no bank, no score, no participant, no model
            Assert.IsFalse(manifest.ContainsKey("banks"));
            Assert.IsFalse(manifest.ContainsKey("participant_id"));
            Assert.IsFalse(manifest.ContainsKey("common_model_id"));
            Assert.IsFalse(manifest.ContainsKey("gate_config_id"));
            Assert.IsFalse(manifest.ContainsKey("query_feature_digest"));
        }

        /// <summary>A quote, a backslash, a newline, a tab, U+001F and DEL in one string.</summary>
        static string HostileDetail()
        {
            var text = new StringBuilder();
            text.Append("quote ").Append('"');
            text.Append(" backslash ").Append('\\');
            text.Append(" newline ").Append('\n');
            text.Append(" tab ").Append('\t');
            text.Append(" unit ").Append((char)0x1F);
            text.Append(" delete ").Append((char)0x7F);
            text.Append(" end");
            return text.ToString();
        }

        [Test]
        public void ARefusalDetailWithHostileCharactersStaysValidJsonAndRoundTrips()
        {
            string detail = HostileDetail();
            FullHandGameCaptureResult refusal = FullHandGameCaptureResult.ForRefusal(
                "capture_escapes", 0, FullHandBridgeCodes.AdaptRefused, detail);

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(refusal));

            string text = ReadText(Manifests()[0]);
            // proves the document actually parses, not merely that a substring appears
            var manifest = (Dictionary<string, object>)TinyJson.Parse(text);
            var failure = (Dictionary<string, object>)manifest["failure"];
            Assert.AreEqual(detail, failure["detail"]);
            Assert.AreEqual(FullHandBridgeCodes.AdaptRefused, failure["code"]);
        }

        [Test]
        public void ANonFiniteValueIsNeitherZeroNorASuccessfulRejection()
        {
            FullHandGameCaptureResult result =
                Outcome("capture_non_finite", "kiwi", SyntheticBinary(10),
                        Pipeline(), UnusableCapture());

            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.None, writer.Record(result));

            var banks = (Dictionary<string, object>)ParsedManifest()["banks"];
            var methods = (List<object>)((Dictionary<string, object>)banks["common_only"])["methods"];
            Assert.AreEqual(FullHandMethods.All.Length, methods.Count);

            for (int index = 0; index < methods.Count; index++)
            {
                var entry = (Dictionary<string, object>)methods[index];
                FullHandMethodDecision decision =
                    result.Outcome.Before.Decision(FullHandMethods.All[index]);
                Assert.IsTrue(double.IsNaN(decision.Quality),
                    "an empty-validity capture must not produce a finite quality");

                // null and flagged, never 0.0 - and never a gate that "rejected" cleanly
                Assert.IsNull(entry["quality"]);
                Assert.AreEqual(false, entry["quality_finite"]);
                Assert.AreEqual(FullHandDecisions.TechnicalFailure, entry["decision"]);
                Assert.AreNotEqual(FullHandDecisions.Abstained, entry["decision"]);
                Assert.AreNotEqual(FullHandDecisions.Accepted, entry["decision"]);
                // the missing evidence itself is recorded rather than scored around
                Assert.IsTrue(entry.ContainsKey("classes_without_evidence"));
            }
        }

        [Test]
        public void TheWrapperRecordsOnceAndReportsASafeCodeOnly()
        {
            var writer = new FullHandCaptureTraceWriter(root);
            FullHandGameCaptureResult original =
                Outcome("capture_wrapped", "kiwi", SyntheticBinary(12), Pipeline(), Capture(2.0));
            var processor = new FullHandGameCaptureTracingProcessor(
                new FixedProcessor(original), writer);

            Assert.AreEqual(FullHandTraceCodes.NotAttempted, processor.LastWriteCode);
            Assert.AreSame(original, processor.Process("capture_wrapped", "kiwi", null));

            Assert.AreEqual(FullHandTraceCodes.None, processor.LastWriteCode);
            Assert.AreEqual(string.Empty, processor.LastWriteDetail);
            Assert.AreEqual(1, processor.RecordCount);
            Assert.AreEqual(1, Manifests().Length);

            // the same result offered again is a duplicate, not a second record
            processor.Process("capture_wrapped", "kiwi", null);
            Assert.AreEqual(FullHandTraceCodes.DuplicateCapture, processor.LastWriteCode);
            Assert.AreEqual(1, Manifests().Length);
            // the diagnostic code carries no path and no capture content
            StringAssert.DoesNotContain(root, processor.LastWriteCode + processor.LastWriteDetail);
        }

        [Test]
        public void ANullResultIsRecordedAsAbsentInsteadOfThrowing()
        {
            var writer = new FullHandCaptureTraceWriter(root);
            Assert.AreEqual(FullHandTraceCodes.ResultAbsent, writer.Record(null));
            Assert.AreEqual(0, Manifests().Length);
        }

        [Test]
        public void NoWriterExistsWhenTheOperatorConfiguredNoDirectory()
        {
            Assert.IsNull(FullHandCaptureTraceWriter.CreateOrNull(null));
            Assert.IsNull(FullHandCaptureTraceWriter.CreateOrNull(string.Empty));
            Assert.IsNull(FullHandCaptureTraceWriter.CreateOrNull("   "));
            Assert.IsNotNull(FullHandCaptureTraceWriter.CreateOrNull(root));
        }

        // ---- the integrated production path ------------------------------------

        [Test]
        public void TheProductionProcessorPersistsTheApprovedAdapterBinaryForOneRawCapture()
        {
            List<RawMediaPipeObservation> observations = RawCapture(6);
            var pipeline = Pipeline();
            var processor = new FullHandGameCaptureProcessor(pipeline, () => Participant, true);
            var writer = new FullHandCaptureTraceWriter(root);
            var traced = new FullHandGameCaptureTracingProcessor(processor, writer);

            FullHandGameCaptureResult result =
                traced.Process("capture_integrated", "kiwi", observations);

            Assert.IsNotNull(result);
            Assert.IsTrue(result.Succeeded, result.FailureCode + " " + result.FailureDetail);
            Assert.AreEqual(1, processor.EvaluationCount, "one capture, exactly one evaluation");
            Assert.AreEqual(observations.Count, result.FrameCount);
            Assert.AreEqual(FullHandTraceCodes.None, traced.LastWriteCode);
            Assert.AreEqual(1, Binaries().Length);
            Assert.AreEqual(1, Manifests().Length);
            Assert.AreEqual(0, StagingFiles().Length);

            // independently rebuild what the APPROVED adapter and writer produce for the same
            // raw observations; the persisted file must be exactly those bytes
            string adaptFailure;
            FullHandCommonCapture capture =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(observations,
                                                                       out adaptFailure);
            Assert.IsNotNull(capture, adaptFailure);
            var timestamps = new double[observations.Count];
            for (int frame = 0; frame < observations.Count; frame++)
                timestamps[frame] = observations[frame].TimestampSeconds;
            byte[] expected = FullHandObservationSession.BuildBinary(capture, timestamps);
            string expectedSha = FullHandObservationSession.Sha256Hex(expected);

            CollectionAssert.AreEqual(expected, File.ReadAllBytes(Binaries()[0]));
            Assert.AreEqual(expectedSha, result.SourceSha256);

            Dictionary<string, object> manifest = ParsedManifest();
            var source = (Dictionary<string, object>)manifest["source_binary"];
            Assert.AreEqual(expectedSha, source["sha256"]);
            Assert.AreEqual(expectedSha, source["source_sha256_from_pipeline"]);
            Assert.AreEqual((double)expected.Length, source["byte_count"]);
            Assert.AreEqual(result.Outcome.QueryFeatureDigest, manifest["query_feature_digest"]);
            Assert.AreEqual(Participant, manifest["participant_id"]);
        }

        [Test]
        public void TheDefaultProductionProcessorRetainsNoBinaryAndWritesNothing()
        {
            List<RawMediaPipeObservation> observations = RawCapture(6);
            // the unchanged two-argument constructor: retention off
            var processor = new FullHandGameCaptureProcessor(Pipeline(), () => Participant);

            FullHandGameCaptureResult result =
                processor.Process("capture_no_retention", "kiwi", observations);

            Assert.IsTrue(result.Succeeded, result.FailureCode + " " + result.FailureDetail);
            Assert.AreEqual(1, processor.EvaluationCount);
            Assert.IsFalse(result.HasSourceBinary, "the default path retains no capture bytes");
            Assert.AreEqual(0, result.SourceBinaryLength);
            Assert.IsNull(result.CopySourceBinary());
            Assert.IsFalse(Directory.Exists(root), "the default path touches no directory");
        }

        // ---- a minimal strict JSON reader, for the tests only -------------------

        /// <summary>
        /// Small strict reader: objects, arrays, strings, numbers, true/false/null. It
        /// REJECTS a raw control character inside a string, so a manifest only parses if the
        /// writer escaped it. Test-only; it is not a general JSON library.
        /// </summary>
        static class TinyJson
        {
            public static object Parse(string text)
            {
                int index = 0;
                object value = ParseValue(text, ref index);
                SkipWhitespace(text, ref index);
                if (index != text.Length) throw new FormatException("trailing content in JSON");
                return value;
            }

            static object ParseValue(string text, ref int index)
            {
                SkipWhitespace(text, ref index);
                if (index >= text.Length) throw new FormatException("unexpected end of JSON");
                char c = text[index];
                if (c == '{') return ParseObject(text, ref index);
                if (c == '[') return ParseArray(text, ref index);
                if (c == '"') return ParseString(text, ref index);
                if (Literal(text, ref index, "true")) return true;
                if (Literal(text, ref index, "false")) return false;
                if (Literal(text, ref index, "null")) return null;
                return ParseNumber(text, ref index);
            }

            static Dictionary<string, object> ParseObject(string text, ref int index)
            {
                var map = new Dictionary<string, object>(StringComparer.Ordinal);
                index++;                                   // '{'
                SkipWhitespace(text, ref index);
                if (index < text.Length && text[index] == '}') { index++; return map; }
                while (true)
                {
                    SkipWhitespace(text, ref index);
                    string name = ParseString(text, ref index);
                    SkipWhitespace(text, ref index);
                    Expect(text, ref index, ':');
                    map[name] = ParseValue(text, ref index);
                    SkipWhitespace(text, ref index);
                    if (index >= text.Length) throw new FormatException("unterminated object");
                    if (text[index] == ',') { index++; continue; }
                    Expect(text, ref index, '}');
                    return map;
                }
            }

            static List<object> ParseArray(string text, ref int index)
            {
                var list = new List<object>();
                index++;                                   // '['
                SkipWhitespace(text, ref index);
                if (index < text.Length && text[index] == ']') { index++; return list; }
                while (true)
                {
                    list.Add(ParseValue(text, ref index));
                    SkipWhitespace(text, ref index);
                    if (index >= text.Length) throw new FormatException("unterminated array");
                    if (text[index] == ',') { index++; continue; }
                    Expect(text, ref index, ']');
                    return list;
                }
            }

            static string ParseString(string text, ref int index)
            {
                Expect(text, ref index, '"');
                var value = new StringBuilder();
                while (true)
                {
                    if (index >= text.Length) throw new FormatException("unterminated string");
                    char c = text[index++];
                    if (c == '"') return value.ToString();
                    if (c < ' ') throw new FormatException("unescaped control character in string");
                    if (c != '\\') { value.Append(c); continue; }
                    if (index >= text.Length) throw new FormatException("dangling escape");
                    char escape = text[index++];
                    if (escape == '"') value.Append('"');
                    else if (escape == '\\') value.Append('\\');
                    else if (escape == '/') value.Append('/');
                    else if (escape == 'b') value.Append('\b');
                    else if (escape == 'f') value.Append('\f');
                    else if (escape == 'n') value.Append('\n');
                    else if (escape == 'r') value.Append('\r');
                    else if (escape == 't') value.Append('\t');
                    else if (escape == 'u')
                    {
                        if (index + 4 > text.Length) throw new FormatException("short unicode escape");
                        value.Append((char)ushort.Parse(text.Substring(index, 4),
                            NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        index += 4;
                    }
                    else throw new FormatException("unknown escape");
                }
            }

            static double ParseNumber(string text, ref int index)
            {
                int start = index;
                while (index < text.Length && "+-.eE0123456789".IndexOf(text[index]) >= 0) index++;
                if (start == index) throw new FormatException("expected a JSON value");
                return double.Parse(text.Substring(start, index - start),
                    NumberStyles.Float, CultureInfo.InvariantCulture);
            }

            static bool Literal(string text, ref int index, string literal)
            {
                if (index + literal.Length > text.Length) return false;
                if (string.CompareOrdinal(text, index, literal, 0, literal.Length) != 0) return false;
                index += literal.Length;
                return true;
            }

            static void Expect(string text, ref int index, char expected)
            {
                if (index >= text.Length || text[index] != expected)
                    throw new FormatException("expected a specific delimiter");
                index++;
            }

            static void SkipWhitespace(string text, ref int index)
            {
                while (index < text.Length && char.IsWhiteSpace(text[index])) index++;
            }
        }
    }
}
