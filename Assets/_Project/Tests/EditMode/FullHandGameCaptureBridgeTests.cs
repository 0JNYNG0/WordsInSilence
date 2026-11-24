// F21 EditMode tests for the full-hand game capture seam.
//
// Everything is synthetic and in memory except (a) isolated temporary files under the OS
// temp directory, created and removed by the test itself, and (b) ONE read of the public
// synthetic bundle fixture at the fixed Validation path, used only to prove that the enabled
// configuration initializes. No webcam, no scene, no private model, no persistentDataPath,
// no network, no environment override of the fixture location and no production cutoff.
// Nothing here has been executed; the operator runs it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandGameCaptureBridgeTests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;
        const int SourceFrames = 4;
        const string Participant = "participant_f21";
        // arbitrary labels; no word rule of any kind is asserted here
        static readonly string[] Words = { "kiwi", "zebra" };

        /// <summary>The fixed public synthetic fixture. Never searched for, never overridden.</summary>
        const string SyntheticFixtureRelative =
            "Validation/m3_full_hand_20260909/f4_synthetic/runtime_manifest.json";

        string temporaryRoot;
        string savedEnable, savedManifest, savedGates;

        [SetUp]
        public void SetUp()
        {
            savedEnable = Environment.GetEnvironmentVariable(
                FullHandGameCaptureConfiguration.EnableVariable);
            savedManifest = Environment.GetEnvironmentVariable(
                FullHandGameCaptureConfiguration.ManifestVariable);
            savedGates = Environment.GetEnvironmentVariable(
                FullHandGameCaptureConfiguration.GatesVariable);
            SetVariables(null, null, null);
            DemoParticipantSession.ResetStaticForTests();
            temporaryRoot = null;
        }

        [TearDown]
        public void TearDown()
        {
            SetVariables(savedEnable, savedManifest, savedGates);
            DemoParticipantSession.ResetStaticForTests();
            if (temporaryRoot != null)
            {
                try { Directory.Delete(temporaryRoot, true); } catch (Exception) { }
                temporaryRoot = null;
            }
        }

        static void SetVariables(string enable, string manifest, string gates)
        {
            Environment.SetEnvironmentVariable(
                FullHandGameCaptureConfiguration.EnableVariable, enable);
            Environment.SetEnvironmentVariable(
                FullHandGameCaptureConfiguration.ManifestVariable, manifest);
            Environment.SetEnvironmentVariable(
                FullHandGameCaptureConfiguration.GatesVariable, gates);
        }

        string TemporaryRoot()
        {
            if (temporaryRoot == null)
            {
                temporaryRoot = Path.Combine(Path.GetTempPath(),
                    "wis_f21_" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(temporaryRoot);
            }
            return temporaryRoot;
        }

        // ---- synthetic raw capture geometry (mirrors the F5 helpers) -----------

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

        // ---- synthetic explicit F20 material (mirrors the F20 helpers) --------

        static FullHandFeatureSequence Sequence(double seed)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            var coordinates = new double[SourceFrames * points * dims];
            var mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = seed + point * 0.13 + frame * 0.05;
                    coordinates[slot * dims + 1] = seed * 0.5 + point * 0.07 - frame * 0.03;
                    coordinates[slot * dims + 2] = 0.0;
                }
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, SourceFrames);
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
            return new FullHandInMemoryCommonBank("model_f21_test_v1", Words,
                new[] { "common_0", "common_1", "common_2", "common_3" },
                new[] { 0, 0, 1, 1 },
                new[] { Sequence(1.0), Sequence(1.4), Sequence(3.0), Sequence(3.4) },
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
        }

        static FullHandGateConfig RejectAllGates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.RejectEverything("gate_reject", FullHandMethods.All[index]);
            return new FullHandGateConfig("config_reject", gates);
        }

        /// <summary>Common-only, EMPTY current-participant additions, exactly as F21 builds it.</summary>
        static FullHandRecognitionPipeline Pipeline()
        {
            return new FullHandRecognitionPipeline(Bank(), RejectAllGates(), Participant,
                FullHandParticipantReferenceSet.Empty(Participant));
        }

        // ---- controller fakes (mirrors the integration-test helpers) ----------

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
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime { }

        sealed class FakeRuntime : IPersonalLiveRecognitionRuntime
        {
            public bool IsPersonalActive { get { return true; } }
            public string ActiveProfileId { get { return "f21_fake_profile"; } }
            public bool ThrowOnRecognize { get; set; }

            public PersonalLiveRoutingResult Recognize(string sessionId,
                RecognitionSequence sequence, string expectedWord)
            {
                if (ThrowOnRecognize)
                    throw new InvalidOperationException("injected legacy recognition failure");
                return SuccessResult(expectedWord);
            }
        }

        // ---- approved synthetic routing success (the existing source helper) --

        static PersonalLiveRoutingResult SuccessResult(string predicted) => Result(
            predicted,
            predicted,
            true,
            true,
            true,
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["GREETING"] = predicted == "GREETING" ? 0.12 : 0.52,
                ["LOVE"] = predicted == "LOVE" ? 0.12 : 0.34,
                ["THANK_YOU"] = predicted == "THANK_YOU" ? 0.12 : 0.56,
            });

        static PersonalLiveRoutingResult Result(
            string predicted,
            string expectedWord,
            bool accepted,
            bool expectedAccept,
            bool questSuccess,
            Dictionary<string, double> scores)
        {
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal)
            {
                ["GREETING"] = false,
                ["LOVE"] = false,
                ["THANK_YOU"] = false,
            };
            accepts[predicted] = accepted;
            accepts[expectedWord] = expectedAccept;
            return new PersonalLiveRoutingResult(
                predicted,
                accepted,
                expectedAccept,
                questSuccess,
                scores,
                accepts,
                scores[predicted],
                "prototype-" + predicted,
                new[] { PersonalRuntimeProfileLoader.GreetingMarginWarning });
        }

        sealed class FakeRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            public FakeRuntime Runtime = new FakeRuntime();
            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
            {
                return new FakeGlobalRuntime();
            }
            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths)
            {
                return Runtime;
            }
        }

        static FakeRuntimeFactory Factory(bool throwOnRecognize)
        {
            var factory = new FakeRuntimeFactory();
            factory.Runtime.ThrowOnRecognize = throwOnRecognize;
            return factory;
        }

        sealed class RecordingProcessor : IFullHandGameCaptureProcessor
        {
            public int Calls;
            public string LastCaptureSessionId;
            public string LastDeclaredTarget;
            public int LastFrameCount = -1;
            public bool Throw;

            public FullHandGameCaptureResult Process(string captureSessionId, string declaredTarget,
                IReadOnlyList<RawMediaPipeObservation> observations)
            {
                Calls++;
                LastCaptureSessionId = captureSessionId;
                LastDeclaredTarget = declaredTarget;
                LastFrameCount = observations == null ? -1 : observations.Count;
                if (Throw) throw new InvalidOperationException("injected processor fault");
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, LastFrameCount,
                    "TEST_EVIDENCE", "synthetic evidence");
            }
        }

        static PersonalLiveRuntimePaths Paths()
        {
            return new PersonalLiveRuntimePaths("templates", "thresholds.json", null, true);
        }

        static PersonalLiveRecognitionController Controller(
            IFullHandGameCaptureProcessor processor, FakeRuntimeFactory factory)
        {
            var controller = new PersonalLiveRecognitionController(
                new FakeObservationSource(), factory, null, false, false, false, processor);
            Assert.IsTrue(controller.Initialize(Paths()), "the fake runtime must initialize");
            return controller;
        }

        const int PreRollFrames = 8;
        const int TrimmedFrames = 6;

        /// <summary>Eight observations at 0.05..0.40s; a 0.06s trailing trim keeps six.</summary>
        static bool RunCapture(PersonalLiveRecognitionController controller)
        {
            Assert.IsTrue(controller.SetExpectedWord("GREETING"));
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(RawCapture(PreRollFrames), "capture_f21"));
            return controller.FinalizeCaptureWithTrailingTrim(0.06d);
        }

        // ---- gate documents ---------------------------------------------------

        /// <summary>
        /// Builds a gate document. <paramref name="thresholds"/> holds the exact STRING form
        /// of each gate's quality_threshold, one per known method; a null entry omits the
        /// field entirely so the "absent" case can be exercised. Every value is an arbitrary
        /// test literal: no production cutoff appears anywhere in this file.
        /// </summary>
        static byte[] GateDocument(string schema, string modelHash, string rejectAll,
                                   string[] thresholds)
        {
            var text = new StringBuilder();
            text.Append("{\"schema\":\"").Append(schema).Append("\",");
            text.Append("\"model_payload_sha256\":\"").Append(modelHash).Append("\",");
            text.Append("\"source_review_sha256\":\"").Append(new string('b', 64)).Append("\",");
            text.Append("\"gates\":[");
            for (int index = 0; index < FullHandMethods.All.Length; index++)
            {
                if (index > 0) text.Append(',');
                text.Append("{\"method\":\"").Append(FullHandMethods.All[index]).Append("\",");
                text.Append("\"reject_all\":\"").Append(rejectAll).Append('"');
                if (thresholds[index] != null)
                    text.Append(",\"quality_threshold\":\"").Append(thresholds[index]).Append('"');
                text.Append('}');
            }
            text.Append("]}");
            return new UTF8Encoding(false).GetBytes(text.ToString());
        }

        static string[] Repeated(string value)
        {
            var values = new string[FullHandMethods.All.Length];
            for (int index = 0; index < values.Length; index++) values[index] = value;
            return values;
        }

        static byte[] RejectAllDocument(string modelHash)
        {
            return GateDocument(FullHandGateDocument.Schema, modelHash, "true",
                                Repeated(FullHandGateDocument.NoThreshold));
        }

        // ---- production bridge -------------------------------------------------

        [Test]
        public void ProductionProcessorReproducesTheSamePipelineResultForTheSameCapture()
        {
            FullHandRecognitionPipeline pipeline = Pipeline();
            var processor = new FullHandGameCaptureProcessor(pipeline, () => Participant);
            List<RawMediaPipeObservation> capture = RawCapture(6);

            FullHandGameCaptureResult bridged = processor.Process("capture_parity", Words[0], capture);
            Assert.IsTrue(bridged.Succeeded, bridged.FailureCode + " " + bridged.FailureDetail);
            Assert.AreEqual(1, processor.EvaluationCount, "one capture is evaluated exactly once");
            Assert.AreEqual(6, bridged.FrameCount);

            // the same input, driven through the same public F20 entry point independently
            string adaptFailure;
            FullHandCommonCapture common =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture, out adaptFailure);
            Assert.IsNotNull(common, adaptFailure);
            var timestamps = new double[capture.Count];
            for (int frame = 0; frame < capture.Count; frame++)
                timestamps[frame] = capture[frame].TimestampSeconds;
            string sourceSha = FullHandObservationSession.Sha256Hex(
                FullHandObservationSession.BuildBinary(common, timestamps));
            FullHandRecognitionOutcome direct =
                Pipeline().Evaluate(common, "capture_parity", sourceSha, Words[0]);

            Assert.AreEqual(direct.QueryFeatureDigest, bridged.Outcome.QueryFeatureDigest);
            Assert.AreEqual(direct.CommonModelId, bridged.Outcome.CommonModelId);
            Assert.AreEqual(direct.ScoringPasses, bridged.Outcome.ScoringPasses);
            foreach (string method in FullHandMethods.All)
            {
                Assert.AreEqual(direct.Before.Decision(method).Score,
                                bridged.Outcome.Before.Decision(method).Score,
                                "score parity for " + method);
                Assert.AreEqual(direct.Before.Decision(method).Decision,
                                bridged.Outcome.Before.Decision(method).Decision,
                                "decision parity for " + method);
            }
        }

        [Test]
        public void AnUnsupportedDeclaredTargetIsEvidenceAndNotABridgeFailure()
        {
            var processor = new FullHandGameCaptureProcessor(Pipeline(), () => Participant);
            // an arbitrary well-formed label that is simply not in this bank's vocabulary
            const string arbitrary = "target-omega_47";

            FullHandGameCaptureResult result =
                processor.Process("capture_target", arbitrary, RawCapture(6));

            Assert.IsTrue(result.Succeeded, result.FailureCode + " " + result.FailureDetail);
            Assert.AreEqual(FullHandBridgeCodes.None, result.FailureCode);
            Assert.AreEqual(arbitrary, result.Outcome.DeclaredTarget,
                            "the declared target must reach F20 exactly as given");
            Assert.IsFalse(result.Outcome.TargetSupported);
            foreach (string method in FullHandMethods.All)
            {
                Assert.AreEqual(FullHandDecisions.UnsupportedTarget,
                    result.Outcome.Before.Decision(method).Decision,
                    "an unsupported target is reported as such, not as undeclared");
                Assert.IsFalse(result.Outcome.Before.Decision(method).TargetSupported);
            }
        }

        [Test]
        public void AMalformedDeclaredTargetRefusesInsteadOfBecomingUndeclared()
        {
            var processor = new FullHandGameCaptureProcessor(Pipeline(), () => Participant);

            // Blank, whitespace-bearing and control-character declarations are all malformed
            // under the F20 identifier contract. The bridge forwards them unchanged, so F20
            // raises its own typed refusal; none of them may be silently downgraded to
            // "no target was declared".
            var malformed = new[]
            {
                string.Empty,
                "   ",
                "target omega",
                // the entry below embeds a literal U+0001 control character
                "targetomega",
            };
            foreach (string declared in malformed)
            {
                FullHandGameCaptureResult result =
                    processor.Process("capture_malformed", declared, RawCapture(5));
                Assert.IsFalse(result.Succeeded,
                               "accepted a malformed declared target: [" + declared + "]");
                Assert.AreEqual(FullHandBridgeCodes.PipelineRefused, result.FailureCode);
                Assert.IsNull(result.Outcome);
            }

            // a genuinely absent declaration stays absent and is still evaluated
            FullHandGameCaptureResult undeclared =
                processor.Process("capture_undeclared", null, RawCapture(5));
            Assert.IsTrue(undeclared.Succeeded,
                          undeclared.FailureCode + " " + undeclared.FailureDetail);
            Assert.IsNull(undeclared.Outcome.DeclaredTarget);
        }

        [Test]
        public void AStaleOrMissingParticipantIdentityRefusesTheCapture()
        {
            string active = Participant;
            var processor = new FullHandGameCaptureProcessor(Pipeline(), () => active);
            Assert.IsTrue(processor.Process("capture_a", Words[0], RawCapture(5)).Succeeded);

            active = "participant_somebody_else";
            FullHandGameCaptureResult stale = processor.Process("capture_b", Words[0], RawCapture(5));
            Assert.IsFalse(stale.Succeeded);
            Assert.AreEqual(FullHandBridgeCodes.ParticipantChanged, stale.FailureCode);

            active = null;
            FullHandGameCaptureResult missing = processor.Process("capture_c", Words[0], RawCapture(5));
            Assert.IsFalse(missing.Succeeded);
            Assert.AreEqual(FullHandBridgeCodes.ParticipantUnavailable, missing.FailureCode);
            Assert.AreEqual(1, processor.EvaluationCount, "a refused capture is never scored");
        }

        // ---- configuration -----------------------------------------------------

        [Test]
        public void TheDisabledConfigurationBuildsNothingAndReadsNothing()
        {
            SetVariables(null, Path.Combine(TemporaryRoot(), "absent_manifest.json"),
                         Path.Combine(TemporaryRoot(), "absent_gates.json"));

            Assert.IsFalse(FullHandGameCaptureConfiguration.IsEnabled());
            Assert.IsNull(FullHandGameCaptureConfiguration.CreateProcessorOrNull());
            Assert.IsFalse(DemoParticipantSession.IsActive,
                           "the disabled path must not initialize a participant");
            Assert.IsFalse(File.Exists(Path.Combine(TemporaryRoot(), "absent_gates.json")));
        }

        [Test]
        public void AnEnabledConfigurationWithoutAnActiveParticipantRefusesVisibly()
        {
            SetVariables(FullHandGameCaptureConfiguration.EnabledValue, "manifest.json",
                         "gates.json");

            IFullHandGameCaptureProcessor processor =
                FullHandGameCaptureConfiguration.CreateProcessorOrNull();
            Assert.IsNotNull(processor, "an enabled invalid configuration is not silently disabled");
            FullHandGameCaptureResult result =
                processor.Process("capture_config", "GREETING", RawCapture(4));
            Assert.IsFalse(result.Succeeded);
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationParticipantMissing, result.FailureCode);
        }

        [Test]
        public void AnEnabledConfigurationWithAMissingGateFileRefusesVisibly()
        {
            string root = TemporaryRoot();
            DemoParticipantSession.Activate(root);
            string manifest = Path.Combine(root, "manifest.json");
            File.WriteAllText(manifest, "{}");
            SetVariables(FullHandGameCaptureConfiguration.EnabledValue, manifest,
                         Path.Combine(root, "absent_gates.json"));

            IFullHandGameCaptureProcessor processor =
                FullHandGameCaptureConfiguration.CreateProcessorOrNull();
            Assert.IsNotNull(processor);
            FullHandGameCaptureResult result =
                processor.Process("capture_config", "GREETING", RawCapture(4));
            Assert.IsFalse(result.Succeeded);
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationPathMissing, result.FailureCode);
        }

        /// <summary>
        /// Initialization only. This proves that the enabled path really assembles a
        /// production processor from the public synthetic bundle and an operator gate
        /// document; it asserts NO accuracy, and the gates written here reject everything.
        /// The private bundle is never searched for or used.
        /// </summary>
        [Test]
        public void TheEnabledConfigurationBuildsAProcessorFromThePublicSyntheticFixture()
        {
            string manifest = Path.Combine(
                Directory.GetParent(Application.dataPath).FullName, SyntheticFixtureRelative);
            FullHandRuntimeBundle bundle = FullHandRuntimeBundle.Load(manifest);
            Assert.IsTrue(bundle.Synthetic,
                          "this test may only ever run against the public synthetic fixture");

            string root = TemporaryRoot();
            DemoParticipantSession.Activate(root);
            string gatesPath = Path.Combine(root, "gates.json");
            byte[] gateBytes = RejectAllDocument(bundle.PayloadSha256);
            File.WriteAllBytes(gatesPath, gateBytes);
            SetVariables(FullHandGameCaptureConfiguration.EnabledValue, manifest, gatesPath);

            var production = FullHandGameCaptureConfiguration.CreateProcessorOrNull()
                as FullHandGameCaptureProcessor;
            Assert.IsNotNull(production,
                             "a valid enabled configuration must build the production processor");
            Assert.AreEqual(DemoParticipantSession.Current.LearnerId, production.BoundParticipantId);

            // no declared target: this exercises initialization, not any word contract
            FullHandGameCaptureResult result =
                production.Process("capture_fixture", null, RawCapture(6));
            Assert.IsTrue(result.Succeeded, result.FailureCode + " " + result.FailureDetail);
            Assert.AreEqual(bundle.PayloadSha256, result.Outcome.CommonModelId);
            Assert.AreEqual(FullHandGateDocument.Sha256Hex(gateBytes), result.Outcome.GateConfigId);
            Assert.AreEqual(1, production.EvaluationCount);
            foreach (string method in FullHandMethods.All)
                Assert.AreNotEqual(FullHandDecisions.Accepted,
                    result.Outcome.Before.Decision(method).Decision,
                    "these gates reject everything, so nothing may be accepted");
        }

        [Test]
        public void AWellFormedGateDocumentParsesWithTheFileBytesAsItsIdentity()
        {
            string modelHash = new string('a', 64);
            byte[] bytes = RejectAllDocument(modelHash);

            FullHandGateConfig config;
            string review, code, detail;
            Assert.IsTrue(FullHandGateDocument.TryParse(bytes, modelHash, out config,
                                                        out review, out code, out detail), detail);
            Assert.AreEqual(FullHandGateDocument.Sha256Hex(bytes), config.ConfigId);
            Assert.AreEqual(new string('b', 64), review);
            foreach (string method in FullHandMethods.All)
                Assert.IsTrue(config.Gate(method).RejectAll);
        }

        [Test]
        public void FiniteStringThresholdsArriveAtTheGatesExactlyAsWritten()
        {
            string modelHash = new string('a', 64);
            // arbitrary test literals in the exact string form the runtime document uses
            var written = new[] { "-1.25", "0", "0.5" };
            var expected = new[] { -1.25, 0.0, 0.5 };
            Assert.AreEqual(written.Length, FullHandMethods.All.Length,
                            "this test writes one threshold per known method");
            byte[] bytes = GateDocument(FullHandGateDocument.Schema, modelHash, "false", written);

            FullHandGateConfig config;
            string review, code, detail;
            Assert.IsTrue(FullHandGateDocument.TryParse(bytes, modelHash, out config,
                                                        out review, out code, out detail), detail);
            for (int index = 0; index < FullHandMethods.All.Length; index++)
            {
                FullHandGate gate = config.Gate(FullHandMethods.All[index]);
                Assert.IsFalse(gate.RejectAll, FullHandMethods.All[index]);
                Assert.AreEqual(expected[index], gate.QualityThreshold,
                                "the parsed threshold must equal the written value exactly");
            }
        }

        [Test]
        public void AMissingOrNonFiniteThresholdIsRejectedAndNeverGuessed()
        {
            string modelHash = new string('a', 64);
            FullHandGateConfig config;
            string review, code, detail;

            // absent field: it must never become a permissive 0.0
            Assert.IsFalse(FullHandGateDocument.TryParse(
                GateDocument(FullHandGateDocument.Schema, modelHash, "false", Repeated(null)),
                modelHash, out config, out review, out code, out detail));
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationGateInvalid, code);
            Assert.IsNull(config, "a missing threshold must never become a permissive gate");

            foreach (string rejected in new[] { "NaN", "Infinity", "-Infinity", "", "abc" })
            {
                Assert.IsFalse(FullHandGateDocument.TryParse(
                    GateDocument(FullHandGateDocument.Schema, modelHash, "false",
                                 Repeated(rejected)),
                    modelHash, out config, out review, out code, out detail),
                    "accepted a non-finite threshold: [" + rejected + "]");
                Assert.AreEqual(FullHandBridgeCodes.ConfigurationGateInvalid, code);
                Assert.IsNull(config);
            }
        }

        [Test]
        public void AnUnknownSchemaOrNonJsonDocumentIsRejected()
        {
            string modelHash = new string('a', 64);
            FullHandGateConfig config;
            string review, code, detail;

            Assert.IsFalse(FullHandGateDocument.TryParse(
                GateDocument("wis_some_other_schema_v9", modelHash, "true",
                             Repeated(FullHandGateDocument.NoThreshold)),
                modelHash, out config, out review, out code, out detail));
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationGateInvalid, code);

            Assert.IsFalse(FullHandGateDocument.TryParse(
                new UTF8Encoding(false).GetBytes("not json at all"), modelHash,
                out config, out review, out code, out detail));
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationGateInvalid, code);
        }

        [Test]
        public void AGateDocumentWrittenForAnotherModelPayloadIsRejected()
        {
            byte[] bytes = RejectAllDocument(new string('a', 64));

            FullHandGateConfig config;
            string review, code, detail;
            Assert.IsFalse(FullHandGateDocument.TryParse(bytes, new string('c', 64), out config,
                                                         out review, out code, out detail));
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationModelHashMismatch, code);
            Assert.IsNull(config);
        }

        // ---- the controller seam ----------------------------------------------

        [Test]
        public void TheTrimmedCaptureReachesTheProcessorExactlyOnceAndTheCaptureCompletes()
        {
            var processor = new RecordingProcessor();
            PersonalLiveRecognitionController controller = Controller(processor, Factory(false));

            Assert.IsTrue(RunCapture(controller), "the ordinary capture must still complete");
            Assert.AreEqual(PersonalLiveCaptureState.Completed, controller.State);
            Assert.IsNotNull(controller.Result);

            Assert.AreEqual(1, processor.Calls);
            Assert.AreEqual("capture_f21", processor.LastCaptureSessionId);
            Assert.AreEqual("GREETING", processor.LastDeclaredTarget);
            Assert.AreEqual(TrimmedFrames, processor.LastFrameCount,
                            "the processor sees the already trailing-trimmed segment");
            Assert.IsNotNull(controller.LastFullHandResult);
        }

        [Test]
        public void FullHandEvidenceSurvivesALaterLegacyFailure()
        {
            var processor = new RecordingProcessor();
            PersonalLiveRecognitionController controller = Controller(processor, Factory(true));

            Assert.IsFalse(RunCapture(controller), "the legacy path fails for this fake runtime");
            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.IsNull(controller.Result);
            Assert.IsNotNull(controller.LastFullHandResult,
                             "the completed bridge result is kept as diagnostic evidence");
            Assert.AreEqual("capture_f21", controller.LastFullHandResult.CaptureSessionId);
        }

        [Test]
        public void LifecycleTransitionsClearTheFullHandResult()
        {
            var processor = new RecordingProcessor();
            PersonalLiveRecognitionController controller = Controller(processor, Factory(false));
            RunCapture(controller);
            Assert.IsNotNull(controller.LastFullHandResult);

            Assert.IsTrue(controller.Reset());
            Assert.IsNull(controller.LastFullHandResult);

            RunCapture(controller);
            Assert.IsNotNull(controller.LastFullHandResult);
            Assert.IsTrue(controller.Reset());
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(RawCapture(5), "capture_f21_again"));
            Assert.IsNull(controller.LastFullHandResult, "beginning a capture clears the evidence");
            Assert.IsTrue(controller.CancelCapture());
            Assert.IsNull(controller.LastFullHandResult);
        }

        [Test]
        public void AThrowingProcessorCannotAlterTheLegacyCaptureOutcome()
        {
            foreach (bool legacyThrows in new[] { false, true })
            {
                PersonalLiveRecognitionController legacy = Controller(null, Factory(legacyThrows));
                bool legacyFinalized = RunCapture(legacy);

                var processor = new RecordingProcessor { Throw = true };
                PersonalLiveRecognitionController bridged =
                    Controller(processor, Factory(legacyThrows));
                bool bridgedFinalized = RunCapture(bridged);

                string context = " (legacyThrows=" + legacyThrows + ")";
                Assert.AreEqual(legacyFinalized, bridgedFinalized, "finalize result" + context);
                Assert.AreEqual(legacy.State, bridged.State, "capture state" + context);
                Assert.AreEqual(legacy.Failure == null, bridged.Failure == null, "failure" + context);
                if (legacy.Failure != null)
                    Assert.AreEqual(legacy.Failure.Code, bridged.Failure.Code, "code" + context);

                if (!legacyThrows)
                {
                    Assert.IsTrue(bridgedFinalized,
                                  "a previously successful capture must still succeed" + context);
                    Assert.AreEqual(PersonalLiveCaptureState.Completed, bridged.State);
                    Assert.IsNotNull(bridged.Result);
                    Assert.AreEqual(legacy.Result.PredictedWord, bridged.Result.PredictedWord);
                    Assert.AreEqual(legacy.Result.Accepted, bridged.Result.Accepted);
                    Assert.AreEqual(legacy.Result.ExpectedWordClassAccept,
                                    bridged.Result.ExpectedWordClassAccept);
                    Assert.AreEqual(legacy.Result.QuestSuccess, bridged.Result.QuestSuccess);
                    Assert.AreEqual(legacy.Result.BestDistance, bridged.Result.BestDistance);
                }

                Assert.IsNotNull(bridged.LastFullHandResult);
                Assert.AreEqual(FullHandBridgeCodes.UnexpectedFailure,
                                bridged.LastFullHandResult.FailureCode, "evidence" + context);
                Assert.IsNull(legacy.LastFullHandResult);
            }
        }

        // ---- the practice attempt ----------------------------------------------

        [Test]
        public void ThePracticeAttemptCarriesTheEvidenceWithoutChangingStrictAuthority()
        {
            FullHandGameCaptureResult evidence =
                FullHandGameCaptureResult.ForRefusal("capture_attempt", 3, "TEST_EVIDENCE", "note");

            PracticeRecognitionAttempt without = Attempt(null);
            PracticeRecognitionAttempt with = Attempt(evidence);

            Assert.IsNull(without.FullHandEvidence);
            Assert.AreSame(evidence, with.FullHandEvidence);
            Assert.AreEqual(without.StrictAccepted, with.StrictAccepted);
            Assert.AreEqual(without.StrictExpectedWordClassAccept, with.StrictExpectedWordClassAccept);
            Assert.AreEqual(without.StrictQuestSuccess, with.StrictQuestSuccess);
            Assert.AreEqual(without.BestDistance, with.BestDistance);
            Assert.AreEqual(without.WinningPrototypeId, with.WinningPrototypeId);
            Assert.AreEqual(without.CommonSignal, with.CommonSignal);
            Assert.IsFalse(with.StrictAccepted, "evidence never grants a strict accept");
        }

        static PracticeRecognitionAttempt Attempt(FullHandGameCaptureResult evidence)
        {
            return new PracticeRecognitionAttempt(
                "capture_attempt", "GREETING", string.Empty,
                PersonalLiveCaptureState.Failed, null, null, null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(), "profile",
                double.NaN, null, null, null, null, evidence);
        }
    }
}
