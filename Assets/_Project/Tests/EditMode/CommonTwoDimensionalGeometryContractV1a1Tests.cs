using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0a1 focused EditMode tests: a single provider's geometry may never
    /// stand in for the other, the whole same-frame pixel-semantic contract must agree,
    /// the contract must stay constant across the capture, and an out-of-contract target
    /// invokes neither the common capture nor the frozen classifier.
    ///
    /// EditMode only: no PlayMode, no webcam, no Scene or Prefab authoring.
    /// </summary>
    public sealed class CommonTwoDimensionalGeometryContractV1a1Tests
    {
        const int Pose = 33, Hand = 21;

        sealed class Geo
        {
            public int W = 1280, H = 720, TexW = 1280, TexH = 720, CamW = 1280, CamH = 720;
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

            public Geo Clone() { return (Geo)MemberwiseClone(); }

            public Step9DInferenceFrameGeometry Build()
            {
                return new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.TestInjected,
                    CamW, CamH, 30, CamW, CamH, Rotation, Mirrored, TexW, TexH, W, H, Format,
                    HFlip, VFlip, PreRotation, Crop, Resize, Letterbox, 0L, 0L, 0, RowOrder);
            }
        }

        static Step9DMergedInferenceGeometry Merged(
            Geo hand, Geo pose, bool handAvailable = true, bool poseAvailable = true)
        {
            Step9DGeometryAcquisitionResult h = handAvailable
                ? Step9DGeometryAcquisitionResult.ForAvailable(hand.Build())
                : Step9DGeometryAcquisitionResult.ForUnavailable(
                    Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady, "hand");
            Step9DGeometryAcquisitionResult p = poseAvailable
                ? Step9DGeometryAcquisitionResult.ForAvailable(pose.Build())
                : Step9DGeometryAcquisitionResult.ForUnavailable(
                    Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady, "pose");
            return new Step9DMergedInferenceGeometry(h, p);
        }

        static RawLandmarkObservationSet Set(int count, int seed, bool isPose)
        {
            var c = new double[count * 3];
            var v = new bool[count];
            for (int i = 0; i < count; i++)
            {
                v[i] = true;
                c[i * 3] = 0.30 + 0.004 * ((i + seed) % 60);
                c[i * 3 + 1] = 0.35 + 0.003 * ((i * 2 + seed) % 60);
                c[i * 3 + 2] = 0.01 * ((i + seed) % 7) - 0.03;
            }
            if (isPose)
            {
                double d = 0.002 * (seed % 5);
                Anchor(c, 11, 0.35 + d, 0.35);
                Anchor(c, 12, 0.65 + d, 0.35);
                Anchor(c, 23, 0.40 + d, 0.68);
                Anchor(c, 24, 0.60 + d, 0.68);
            }
            return new RawLandmarkObservationSet(count, c, v, new bool[count], new bool[count]);
        }

        static void Anchor(double[] c, int lm, double x, double y)
        {
            c[lm * 3] = x; c[lm * 3 + 1] = y; c[lm * 3 + 2] = 0.0;
        }

        static List<RawMediaPipeObservation> Capture(
            int frames, Func<int, Step9DMergedInferenceGeometry> geometry)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int f = 0; f < frames; f++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (f + 1), RawObservationConvention.LiveMediaPipe,
                    Set(Pose, f, true), Set(Hand, f + 11, false), Set(Hand, f + 23, false),
                    geometry(f)));
            return list;
        }

        static string FailureFor(List<RawMediaPipeObservation> capture)
        {
            string failure;
            CommonTwoDimensionalCapture c =
                ApprovedLivePreprocessingAdapter.AdaptCommon2D(capture, out failure);
            Assert.IsNull(c, "this capture must fail closed");
            Assert.IsFalse(string.IsNullOrEmpty(failure), "a concrete detail is required");
            return failure;
        }

        [Test]
        public void BothProviderGeometriesCompatibleSucceeds()
        {
            var g = new Geo();
            string failure;
            CommonTwoDimensionalCapture c = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(10, f => Merged(g.Clone(), g.Clone())), out failure);
            Assert.IsNotNull(c, failure);
            Assert.AreEqual(1280, c.ImageWidth);
            Assert.AreEqual(720, c.ImageHeight);
        }

        [Test]
        public void ASingleProviderGeometryIsNeverRepresentative()
        {
            var g = new Geo();
            StringAssert.Contains("both provider",
                FailureFor(Capture(10, f => Merged(g.Clone(), g.Clone(), false, true))));
            StringAssert.Contains("both provider",
                FailureFor(Capture(10, f => Merged(g.Clone(), g.Clone(), true, false))));
            StringAssert.Contains("both provider",
                FailureFor(Capture(10, f => Merged(g.Clone(), g.Clone(), false, false))));
        }

        [Test]
        public void SameFramePixelSemanticMismatchFailsClosed()
        {
            var cases = new Dictionary<string, Action<Geo>>
            {
                { "inference_image_width", g => { g.W = 640; g.H = 480; } },
                { "inference_texture_width", g => g.TexW = 800 },
                { "observed_webcam_width", g => g.CamW = 800 },
                { "image_format", g => g.Format = "BGRA32" },
                { "pre_inference_horizontal_flip",
                    g => g.HFlip = Step9DPreInferenceTransformState.Applied },
                { "pre_inference_vertical_flip",
                    g => g.VFlip = Step9DPreInferenceTransformState.Applied },
                { "crop", g => g.Crop = Step9DPreInferenceTransformState.Applied },
                { "resize", g => g.Resize = Step9DPreInferenceTransformState.Applied },
                { "letterbox", g => g.Letterbox = Step9DPreInferenceTransformState.Applied },
                { "pixel_buffer_row_order", g => g.RowOrder = "BottomUp" },
                { "pre_inference_rotation_degrees", g => g.PreRotation = 90 },
                { "observed_video_rotation_angle", g => g.Rotation = 180 },
                { "observed_video_vertically_mirrored", g => g.Mirrored = true },
            };
            foreach (KeyValuePair<string, Action<Geo>> c in cases)
            {
                var hand = new Geo();
                c.Value(hand);
                string failure = FailureFor(Capture(10, f => Merged(hand, new Geo())));
                StringAssert.Contains("contract mismatch", failure);
                StringAssert.Contains(c.Key, failure);
            }
        }

        [Test]
        public void KnownUnknownNullableStateIsNotTreatedAsCompatible()
        {
            var hand = new Geo();
            hand.PreRotation = null;
            StringAssert.Contains("known/unknown",
                FailureFor(Capture(10, f => Merged(hand, new Geo()))));

            var mirror = new Geo();
            mirror.Mirrored = null;
            StringAssert.Contains("known/unknown",
                FailureFor(Capture(10, f => Merged(mirror, new Geo()))));
        }

        [Test]
        public void CaptureWideContractDriftFailsClosed()
        {
            var drift = new Geo();
            drift.W = 640;
            drift.H = 480;
            string failure = FailureFor(Capture(10,
                f => f == 5 ? Merged(drift, drift) : Merged(new Geo(), new Geo())));
            StringAssert.Contains("changed within the capture", failure);
            StringAssert.Contains("observation 5", failure);

            var flip = new Geo();
            flip.HFlip = Step9DPreInferenceTransformState.Applied;
            StringAssert.Contains("changed within the capture", FailureFor(Capture(10,
                f => f == 7 ? Merged(flip, flip) : Merged(new Geo(), new Geo()))));
        }

        [Test]
        public void OutOfContractTargetInvokesNeitherCaptureNorClassifier()
        {
            var spy = new CountingClassifier();
            foreach (string target in new[] { "LOVE", "", null, "WORD1290" })
            {
                spy.Calls = 0;
                Assert.IsFalse(FrozenCommonSignModel.IsSupportedTarget(target));
                CommonEvidence evidence = CommonEvidence.From(null, target);
                Assert.AreEqual(CommonEvidenceState.NotApplicable, evidence.State);
                Assert.AreEqual(CommonSignFailureReason.None, evidence.FailureReason);
                Assert.AreEqual(0, spy.Calls);
            }
            foreach (string target in new[] { "GREETING", "THANK_YOU" })
                Assert.IsTrue(FrozenCommonSignModel.IsSupportedTarget(target));
        }

        sealed class CountingClassifier : ICommonSignClassifier
        {
            public int Calls;
            public string ArtifactVersion { get { return "test"; } }
            public string ArtifactSha256 { get { return "test"; } }
            public CommonSignClassification Classify(
                double[] coordinates, bool[] validMask, int frameCount, string requestedTarget)
            {
                Calls++;
                return CommonSignClassification.Failed("test", "test", requestedTarget,
                    CommonSignFailureReason.SequenceMissing, "spy");
            }
        }
    }
}
