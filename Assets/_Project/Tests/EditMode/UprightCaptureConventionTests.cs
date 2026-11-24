// F51-R1 EditMode tests for the capture-convention integration defect.
//
// Synthetic observations only: no webcam, scene, model, private fixture or recording. The
// two private controller gates are reached by reflection so no runtime or model fixture has
// to be loaded. Nothing here asserts recognition accuracy or any performance property.
using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class UprightCaptureConventionTests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;

        // ---- fixtures ----------------------------------------------------------

        static Step9DInferenceFrameGeometry Geometry(bool upright, int? rotation, bool? mirrored)
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                Width, Height, 30, Width, Height, rotation, mirrored,
                Width, Height, Width, Height, "RGBA32",
                Step9DPreInferenceTransformState.None,
                upright ? Step9DPreInferenceTransformState.Applied
                        : Step9DPreInferenceTransformState.None,
                0, Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, 0L, 0, upright ? "top-down" : "bottom-up");
        }

        static Step9DMergedInferenceGeometry Merged(bool handUpright, bool poseUpright,
                                                    int? rotation = 0, bool? mirrored = false)
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(
                    Geometry(handUpright, rotation, mirrored)),
                Step9DGeometryAcquisitionResult.ForAvailable(
                    Geometry(poseUpright, rotation, mirrored)));
        }

        static double[] NaNs(int count)
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

        /// <summary>Pose fixture in y-down image coordinates; flip mirrors it to y-up.</summary>
        static RawLandmarkObservationSet PoseSet(int frame, bool flip)
        {
            var c = NaNs(Pose);
            var v = new bool[Pose];
            double drift = 0.002 * frame;
            var rows = new[] { new[] { 11.0, 0.35, 0.35 }, new[] { 12.0, 0.65, 0.35 },
                               new[] { 23.0, 0.40, 0.68 }, new[] { 24.0, 0.60, 0.68 },
                               new[] { 13.0, 0.30, 0.45 }, new[] { 14.0, 0.70, 0.45 },
                               new[] { 15.0, 0.28, 0.55 }, new[] { 16.0, 0.72, 0.55 } };
            foreach (double[] row in rows)
                Put(c, v, (int)row[0], row[1] + drift, flip ? 1.0 - row[2] : row[2]);
            return new RawLandmarkObservationSet(Pose, c, v, new bool[Pose], new bool[Pose]);
        }

        static RawLandmarkObservationSet HandSet(int frame, double baseX, double baseY, bool flip)
        {
            var c = NaNs(Hand);
            var v = new bool[Hand];
            for (int joint = 0; joint < Hand; joint++)
            {
                double y = baseY + 0.003 * joint - 0.0005 * frame;
                Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame, flip ? 1.0 - y : y);
            }
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], new bool[Hand]);
        }

        static RawMediaPipeObservation Observation(int frame, RawObservationConvention convention,
                                                   Step9DMergedInferenceGeometry geometry,
                                                   bool flip = false)
        {
            return new RawMediaPipeObservation(
                0.05 * (frame + 1), convention, PoseSet(frame, flip),
                HandSet(frame, 0.30, 0.50, flip), HandSet(frame, 0.60, 0.50, flip), geometry);
        }

        static RawObservationConvention Upright
        {
            get { return RawObservationConvention.LiveMediaPipeUprightTexture; }
        }

        // ---- controller access -------------------------------------------------

        static PersonalLiveRecognitionController Controller()
        {
            return new PersonalLiveRecognitionController(null);
        }

        static MethodInfo Method(string name)
        {
            MethodInfo method = typeof(PersonalLiveRecognitionController).GetMethod(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(method, name + " must exist on the controller");
            return method;
        }

        static Exception PreRoll(PersonalLiveRecognitionController controller,
                                 params RawMediaPipeObservation[] preRoll)
        {
            try
            {
                Method("ValidatePreRoll").Invoke(controller, new object[] { preRoll });
                return null;
            }
            catch (TargetInvocationException error) { return error.InnerException; }
        }

        static bool Append(PersonalLiveRecognitionController controller,
                           RawMediaPipeObservation observation, out string rejection)
        {
            var arguments = new object[] { observation, null };
            bool accepted = (bool)Method("TryAppendObservation").Invoke(controller, arguments);
            rejection = arguments[1] as string;
            return accepted;
        }

        // ---- tests -------------------------------------------------------------

        [Test]
        public void LegacyConventionKeepsItsExistingAcceptance()
        {
            var legacy = RawObservationConvention.LiveMediaPipe;
            var controller = Controller();
            Assert.IsNull(PreRoll(controller, Observation(0, legacy, Merged(false, false)),
                                  Observation(1, legacy, Merged(false, false))));
            string rejection;
            Assert.IsTrue(Append(controller, Observation(0, legacy, Merged(false, false)),
                                 out rejection), rejection);
            Assert.IsTrue(Append(controller, Observation(1, legacy, null), out rejection),
                          "legacy acceptance does not start requiring geometry");
        }

        [Test]
        public void UprightConventionWithVerifiedGeometryIsAcceptedByBothGates()
        {
            var controller = Controller();
            Assert.IsNull(PreRoll(controller, Observation(0, Upright, Merged(true, true)),
                                  Observation(1, Upright, Merged(true, true))));
            string rejection;
            Assert.IsTrue(Append(controller, Observation(0, Upright, Merged(true, true)),
                                 out rejection), rejection);
            Assert.IsTrue(Append(controller, Observation(1, Upright, Merged(true, true)),
                                 out rejection), rejection);
        }

        [TestCase("missing")]      // no merged geometry at all
        [TestCase("mixed")]        // hand upright, pose not
        [TestCase("legacyMeta")]   // upright convention but bottom-up metadata
        [TestCase("rotated")]      // unsupported camera rotation
        [TestCase("mirrored")]     // videoVerticallyMirrored true
        [TestCase("unknown")]      // rotation/mirror not reported
        public void UprightWithoutVerifiedGeometryIsRefused(string mode)
        {
            Step9DMergedInferenceGeometry geometry;
            switch (mode)
            {
                case "missing": geometry = null; break;
                case "mixed": geometry = Merged(true, false); break;
                case "legacyMeta": geometry = Merged(false, false); break;
                case "rotated": geometry = Merged(true, true, 90, false); break;
                case "mirrored": geometry = Merged(true, true, 0, true); break;
                default: geometry = Merged(true, true, null, null); break;
            }
            var controller = Controller();
            Assert.IsInstanceOf<ArgumentException>(
                PreRoll(controller, Observation(0, Upright, geometry)));
            string rejection;
            Assert.IsFalse(Append(controller, Observation(0, Upright, geometry), out rejection));
            Assert.IsFalse(string.IsNullOrEmpty(rejection), "a concrete reason is required");
        }

        [Test]
        public void AnArbitraryNormalizedImageConventionIsNeverAccepted()
        {
            var foreign = new RawObservationConvention(
                "some_other_normalized_image_v9",
                RawObservationCoordinateSpace.MediaPipeNormalizedImage,
                "x-right/y-up in source texture grid", "direct", "not mirrored");
            var controller = Controller();
            Assert.IsInstanceOf<ArgumentException>(
                PreRoll(controller, Observation(0, foreign, Merged(true, true))));
            string rejection;
            Assert.IsFalse(Append(controller, Observation(0, foreign, Merged(true, true)),
                                  out rejection));
        }

        [Test]
        public void LegacyAndUprightCannotBeMixedInsideOneCapture()
        {
            var legacy = RawObservationConvention.LiveMediaPipe;
            Assert.IsInstanceOf<ArgumentException>(PreRoll(Controller(),
                Observation(0, legacy, Merged(false, false)),
                Observation(1, Upright, Merged(true, true))));

            var controller = Controller();
            string rejection;
            Assert.IsTrue(Append(controller, Observation(0, legacy, Merged(false, false)),
                                 out rejection), rejection);
            Assert.IsFalse(Append(controller, Observation(1, Upright, Merged(true, true)),
                                  out rejection), "the capture convention must stay stable");
            StringAssert.Contains("changed", rejection);
        }

        [Test]
        public void UprightSourceTextureNormalizationMatchesTheEquivalentLegacyScene()
        {
            var legacyCapture = new List<RawMediaPipeObservation>();
            var uprightCapture = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < 3; frame++)
            {
                legacyCapture.Add(Observation(frame, RawObservationConvention.LiveMediaPipe,
                                              Merged(false, false)));
                uprightCapture.Add(Observation(frame, Upright, Merged(true, true), true));
            }
            string legacyFailure, uprightFailure;
            FullHandCommonCapture legacy = ApprovedLivePreprocessingAdapter
                .AdaptFullHandCommon2D(legacyCapture, out legacyFailure);
            FullHandCommonCapture upright = ApprovedLivePreprocessingAdapter
                .AdaptFullHandCommon2D(uprightCapture, out uprightFailure);
            Assert.IsNotNull(legacy, legacyFailure);
            Assert.IsNotNull(upright, uprightFailure);
            Assert.AreEqual(legacy.FrameCount, upright.FrameCount);

            int compared = 0;
            for (int frame = 0; frame < legacy.FrameCount; frame++)
                for (int point = 0; point < 46; point++)
                {
                    string where = " frame " + frame + " point " + point;
                    Assert.AreEqual(legacy.IsValid(frame, point), upright.IsValid(frame, point),
                                    "mask differs at" + where);
                    if (!legacy.IsValid(frame, point)) continue;
                    // the torso frame is data-derived, so a global y reflection of the whole
                    // scene reproduces the same normalized point; z stays exactly 0
                    Assert.AreEqual(legacy.Coordinate(frame, point, 0),
                                    upright.Coordinate(frame, point, 0), 1e-9, "x at" + where);
                    Assert.AreEqual(legacy.Coordinate(frame, point, 1),
                                    upright.Coordinate(frame, point, 1), 1e-9, "y at" + where);
                    Assert.AreEqual(0.0, upright.Coordinate(frame, point, 2), 1e-12,
                                    "z at" + where);
                    compared++;
                }
            Assert.AreEqual(3 * 46, compared, "all 46 points of every frame are compared");
        }
    }
}
