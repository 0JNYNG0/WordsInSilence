using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0a focused EditMode tests for the ACTIVE common-2D domain and the
    /// conservative fusion consumption. EditMode only: no PlayMode, no webcam, no Scene
    /// or Prefab authoring.
    ///
    /// The numeric replay against the approved retained captures and the authoritative
    /// torso_normalize_2d reference lives in the offline Delivery harness, because the
    /// retained traces are deliberately not installed as Unity build assets.
    /// </summary>
    public sealed class FrozenCommonSignActiveDomainV1aTests
    {
        const int Pose = 33, Hand = 21;

        static Step9DMergedInferenceGeometry Merged(int w, int h)
        {
            var g = new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                w, h, 30, w, h, 0, false, w, h, w, h, "RGBA32",
                Step9DPreInferenceTransformState.None, Step9DPreInferenceTransformState.None,
                0, Step9DPreInferenceTransformState.None, Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None, 0L, 0L, 0, "TopDown");
            var r = Step9DGeometryAcquisitionResult.ForAvailable(g);
            return new Step9DMergedInferenceGeometry(r, r);
        }

        static RawLandmarkObservationSet Set(int count, int seed, double zScale,
            bool isPose = false)
        {
            var c = new double[count * 3];
            var v = new bool[count];
            var re = new bool[count];
            var st = new bool[count];
            for (int i = 0; i < count; i++)
            {
                v[i] = true;
                c[i * 3] = 0.30 + 0.004 * ((i + seed) % 60);
                c[i * 3 + 1] = 0.35 + 0.003 * ((i * 2 + seed) % 60);
                c[i * 3 + 2] = zScale * (0.01 * ((i + seed) % 7) - 0.03);
            }
            if (isPose)
            {
                // Anatomically sane torso anchors: without them the approved
                // shoulder/hip basis degenerates and every landmark is masked out.
                double drift = 0.002 * (seed % 5);
                Anchor(c, 11, 0.35 + drift, 0.35);
                Anchor(c, 12, 0.65 + drift, 0.35);
                Anchor(c, 23, 0.40 + drift, 0.68);
                Anchor(c, 24, 0.60 + drift, 0.68);
            }
            return new RawLandmarkObservationSet(count, c, v, re, st);
        }

        static void Anchor(double[] c, int landmark, double x, double y)
        {
            c[landmark * 3] = x;
            c[landmark * 3 + 1] = y;
            c[landmark * 3 + 2] = 0.0;
        }

        static List<RawMediaPipeObservation> Capture(
            int frames, Step9DMergedInferenceGeometry geometry, double zScale = 1.0)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int f = 0; f < frames; f++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (f + 1), RawObservationConvention.LiveMediaPipe,
                    Set(Pose, f, zScale, true), Set(Hand, f + 11, zScale),
                    Set(Hand, f + 23, zScale), geometry));
            return list;
        }

        // ---- active common-2D domain ---------------------------------------

        [Test]
        public void CommonCaptureIsBuiltFromRawPostGapWithZeroDepth()
        {
            string failure;
            CommonTwoDimensionalCapture capture = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(12, Merged(1280, 720)), out failure);
            Assert.IsNotNull(capture, failure);
            Assert.AreEqual(12, capture.FrameCount);
            Assert.AreEqual(16, capture.LandmarkCount);
            Assert.AreEqual(1280, capture.ImageWidth);
            Assert.AreEqual(720, capture.ImageHeight);
            Assert.AreEqual("raw_post_gap_pixel_plane_z0_torso_normalize_2d", capture.DomainContract);

            double[] coordinates = capture.CopyCoordinates();
            for (int i = 2; i < coordinates.Length; i += 3)
                Assert.AreEqual(0.0, coordinates[i], 0.0,
                    "the common-2D contract fixes channel 2 at exactly zero");
        }

        [Test]
        public void CommonCaptureIsInvariantToThePseudoDepthChannel()
        {
            string f1, f2;
            double[] a = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(12, Merged(1280, 720), 1.0), out f1).CopyCoordinates();
            double[] b = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(12, Merged(1280, 720), -17.5), out f2).CopyCoordinates();
            CollectionAssert.AreEqual(a, b,
                "pseudo-z must not reach the common-2D transform at all");
        }

        [Test]
        public void CommonCaptureActuallyConsumesTheImageGeometry()
        {
            string f1, f2;
            double[] a = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(12, Merged(1280, 720)), out f1).CopyCoordinates();
            double[] b = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(12, Merged(640, 480)), out f2).CopyCoordinates();
            Assert.IsFalse(a.SequenceEqual(b),
                "a different pixel plane must produce a different common-2D domain");
        }

        [Test]
        public void MissingOrInconsistentImageGeometryFailsClosed()
        {
            string failure;
            Assert.IsNull(ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                Capture(8, null), out failure));
            Assert.IsNotNull(failure);

            List<RawMediaPipeObservation> mixed = Capture(8, Merged(1280, 720));
            mixed[4] = new RawMediaPipeObservation(
                mixed[4].TimestampSeconds, RawObservationConvention.LiveMediaPipe,
                mixed[4].Pose, mixed[4].LeftHand, mixed[4].RightHand, Merged(640, 480));
            Assert.IsNull(ApprovedLivePreprocessingAdapter.AdaptCommon2D(mixed, out failure));
            StringAssert.Contains("geometry changed", failure);

            Assert.IsNull(ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                new List<RawMediaPipeObservation>(), out failure));
        }

        [Test]
        public void CommonCaptureIsNotThePseudoThreeDimensionalSequence()
        {
            var observations = Capture(12, Merged(1280, 720));
            string failure;
            double[] common = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                observations, out failure).CopyCoordinates();
            ApprovedPreprocessingResult pseudo =
                new ApprovedLivePreprocessingAdapter().Adapt(observations);

            bool anyPseudoDepth = false;
            for (int frame = 0; frame < pseudo.NormalizedFrameCount && !anyPseudoDepth; frame++)
            {
                double[] c = pseudo.Frames[frame].CopyCoordinates();
                for (int i = 2; i < c.Length; i += 3)
                    if (Math.Abs(c[i]) > 1e-9) { anyPseudoDepth = true; break; }
            }
            Assert.IsTrue(anyPseudoDepth,
                "the strict/personal route keeps a non-zero pseudo-depth channel");
            Assert.AreEqual(0.0, common[2], 0.0,
                "the common route must not inherit that pseudo-depth basis");
        }

        // ---- conservative fusion consumption --------------------------------

        static GuidedQuestOutcome Route(string expected, string predicted, bool questSuccess,
            CommonSignClassification common,
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Completed)
        {
            return GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate, state, true,
                expected, predicted, questSuccess, questSuccess, questSuccess,
                false, false, common);
        }

        static CommonSignClassification Eligible(string target, bool support)
        {
            bool greetingTop = (target == "GREETING") == support;
            double g = greetingTop ? 0.9 : 0.1;
            return CommonSignClassification.Success(
                "common_2d_v1_6", "sha", target, g, 1.0 - g, greetingTop ? 2.2 : -2.2, 32);
        }

        [Test]
        public void OutcomeSeamConsumesTheCommonInput()
        {
            Assert.AreEqual(CommonEvidenceState.Unavailable,
                Route("GREETING", "GREETING", true, null).Common.State);
            Assert.AreEqual(CommonEvidenceState.Support,
                Route("GREETING", "GREETING", true, Eligible("GREETING", true)).Common.State);
            Assert.AreEqual(CommonEvidenceState.NonSupport,
                Route("GREETING", "GREETING", true, Eligible("GREETING", false)).Common.State);
            Assert.AreEqual(CommonEvidenceState.TechnicalEvidence,
                Route("GREETING", "GREETING", true, CommonSignClassification.Failed(
                    "common_2d_v1_6", "sha", "GREETING",
                    CommonSignFailureReason.AllLandmarksMissing, "none")).Common.State);
            Assert.AreEqual(CommonEvidenceState.NotApplicable,
                Route("LOVE", "LOVE", true, Eligible("GREETING", true)).Common.State);
        }

        [Test]
        public void CommonEvidenceNeverChangesTheStrictOutcome()
        {
            var shapes = new[]
            {
                new object[] { "GREETING", "GREETING", true, PersonalLiveCaptureState.Completed },
                new object[] { "GREETING", "GREETING", false, PersonalLiveCaptureState.Completed },
                new object[] { "GREETING", "THANK_YOU", false, PersonalLiveCaptureState.Completed },
                new object[] { "GREETING", "GREETING", true, PersonalLiveCaptureState.Failed },
                new object[] { "LOVE", "LOVE", true, PersonalLiveCaptureState.Completed },
            };
            var commons = new List<CommonSignClassification>
            {
                null, Eligible("GREETING", true), Eligible("GREETING", false),
                Eligible("THANK_YOU", true),
                CommonSignClassification.Failed("common_2d_v1_6", "sha", "GREETING",
                    CommonSignFailureReason.NonFiniteInput, "nan"),
            };
            foreach (object[] s in shapes)
            {
                GuidedQuestOutcomeKind baseline = Route(
                    (string)s[0], (string)s[1], (bool)s[2], null,
                    (PersonalLiveCaptureState)s[3]).Kind;
                foreach (CommonSignClassification c in commons)
                    Assert.AreEqual(baseline, Route((string)s[0], (string)s[1], (bool)s[2], c,
                        (PersonalLiveCaptureState)s[3]).Kind,
                        "common evidence must never move the strict outcome");
            }
        }

        [Test]
        public void CommonEvidenceCarriesNoAcceptanceAuthority()
        {
            GuidedQuestOutcome o = Route("GREETING", "GREETING", true, Eligible("GREETING", true));
            Assert.IsFalse(o.Common.GrantsSuccess);
            Assert.IsFalse(o.Common.GrantsLearnerError);
            Assert.IsFalse(o.Common.GrantsVeto);

            string[] forbidden = { "Write", "Save", "Activate", "Complete", "Repository", "Quest" };
            string[] members = typeof(CommonEvidence)
                .GetMembers(BindingFlags.Public | BindingFlags.Instance).Select(m => m.Name).ToArray();
            foreach (string bad in forbidden)
                Assert.IsFalse(members.Any(m => m.IndexOf(bad, StringComparison.Ordinal) >= 0),
                    "common evidence must not expose " + bad);
        }

        [Test]
        public void LegacyLoveRouteIsUnaffected()
        {
            GuidedQuestOutcome love = Route("LOVE", "LOVE", true, Eligible("GREETING", true));
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, love.Kind);
            Assert.AreEqual(CommonEvidenceState.NotApplicable, love.Common.State);
            Assert.AreEqual(CommonSignFailureReason.None, love.Common.FailureReason);
        }

        [Test]
        public void SnapshotExposesTheCommonCaptureSeamForBothRoutes()
        {
            Assert.IsNotNull(typeof(PersonalLiveRecognitionSnapshot)
                .GetProperty("Common2DCapture", BindingFlags.Public | BindingFlags.Instance));
            Assert.IsNotNull(typeof(PersonalLiveRecognitionSnapshot)
                .GetProperty("Common2DFailure", BindingFlags.Public | BindingFlags.Instance));
            Assert.IsNotNull(typeof(GuidedQuestOutcome)
                .GetProperty("Common", BindingFlags.Public | BindingFlags.Instance));
        }
    }
}
