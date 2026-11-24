// F2 EditMode tests for the 46-point full-hand common-2D candidate.
//
// No webcam, no scene, no file, no private data and no model. The 16-point comparisons use
// synthetic pixel coordinates that both paths see identically; no claim is made about any
// external corpus. Existing test classes are untouched.
using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandCommonCaptureTests
    {
        const int Pose = 33, Hand = 21, Points = 46, Legacy = 16;
        const int Width = 1280, Height = 720;
        const int LeftBase = 4, RightBase = 25;

        // ---- geometry fixture, same shape as the approved contract tests -------

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

        static Step9DMergedInferenceGeometry Merged(Geo hand, Geo pose)
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(hand.Build()),
                Step9DGeometryAcquisitionResult.ForAvailable(pose.Build()));
        }

        static Step9DMergedInferenceGeometry Standard() { return Merged(new Geo(), new Geo()); }

        // ---- landmark fixtures -------------------------------------------------

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
            Put(c, v, 11, 0.35 + drift, 0.35);      // anchors
            Put(c, v, 12, 0.65 + drift, 0.35);
            Put(c, v, 23, 0.40 + drift, 0.68);
            Put(c, v, 24, 0.60 + drift, 0.68);
            Put(c, v, 13, 0.30 + drift, 0.45);      // emitted pose points
            Put(c, v, 14, 0.70 + drift, 0.45);
            Put(c, v, 15, 0.28 + drift, 0.55);
            Put(c, v, 16, 0.72 + drift, 0.55);
            return new RawLandmarkObservationSet(Pose, c, v, new bool[Pose], new bool[Pose]);
        }

        static RawLandmarkObservationSet HandSet(int frame, double baseX, double baseY,
            int missingJoint = -1, bool strongAtMissing = false,
            IDictionary<int, double[]> overrides = null)
        {
            var c = NaNArray(Hand);
            var v = new bool[Hand];
            var strong = new bool[Hand];
            for (int joint = 0; joint < Hand; joint++)
            {
                if (joint == missingJoint)
                {
                    strong[joint] = strongAtMissing;
                    continue;
                }
                Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame,
                                 baseY + 0.003 * joint - 0.0005 * frame);
            }
            if (overrides != null)
                foreach (KeyValuePair<int, double[]> pair in overrides)
                    Put(c, v, pair.Key, pair.Value[0], pair.Value[1]);
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], strong);
        }

        static List<RawMediaPipeObservation> Capture(
            int frames,
            Func<int, RawLandmarkObservationSet> left = null,
            Func<int, Step9DMergedInferenceGeometry> geometry = null,
            RawObservationConvention convention = null,
            Func<int, double> timestamp = null)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    timestamp == null ? 0.05 * (frame + 1) : timestamp(frame),
                    convention ?? RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame),
                    left == null ? HandSet(frame, 0.30, 0.50) : left(frame),
                    HandSet(frame, 0.60, 0.50),
                    geometry == null ? Standard() : geometry(frame)));
            return list;
        }

        static FullHandCommonCapture Adapted(IReadOnlyList<RawMediaPipeObservation> capture)
        {
            string failure;
            FullHandCommonCapture result =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture, out failure);
            Assert.IsNotNull(result, failure);
            return result;
        }

        static void Refused(IReadOnlyList<RawMediaPipeObservation> capture, string expected)
        {
            string failure;
            FullHandCommonCapture result =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(capture, out failure);
            Assert.IsNull(result, "this capture must fail closed");
            Assert.IsFalse(string.IsNullOrEmpty(failure), "a concrete reason is required");
            if (expected != null) StringAssert.Contains(expected, failure);
        }

        /// <summary>
        /// Masks must match everywhere; coordinates must match exactly wherever the point
        /// is valid. A masked slot is deliberately NOT compared: the frozen capture zeroes
        /// it while the candidate keeps NaN, which is the accepted masking difference.
        /// </summary>
        static void AssertLegacyProjectionMatches(IReadOnlyList<RawMediaPipeObservation> capture)
        {
            FullHandCommonCapture full = Adapted(capture);
            string failure;
            CommonTwoDimensionalCapture legacy =
                ApprovedLivePreprocessingAdapter.AdaptCommon2D(capture, out failure);
            Assert.IsNotNull(legacy, failure);
            Assert.AreEqual(legacy.FrameCount, full.FrameCount);

            int[] projection = FullHandCommonSchema.CopyLegacy16PointIndices();
            double[] legacyCoordinates = legacy.CopyCoordinates();
            bool[] legacyMask = legacy.CopyValidMask();
            for (int frame = 0; frame < full.FrameCount; frame++)
                for (int slot = 0; slot < Legacy; slot++)
                {
                    int legacySlot = frame * Legacy + slot;
                    string where = " at frame " + frame + " legacy slot " + slot;
                    Assert.AreEqual(legacyMask[legacySlot],
                                    full.IsValid(frame, projection[slot]), "mask differs" + where);
                    if (!legacyMask[legacySlot]) continue;
                    for (int axis = 0; axis < 3; axis++)
                        Assert.AreEqual(legacyCoordinates[legacySlot * 3 + axis],
                                        full.Coordinate(frame, projection[slot], axis), 0.0,
                                        "coordinate differs" + where);
                }
        }

        // ==== schema ============================================================

        [Test]
        public void PointSchemaIsFortySixPointsInTheFrozenOrder()
        {
            Assert.AreEqual("wis_full_hand_common_v1", FullHandCommonSchema.SchemaId);
            FullHandPointSpec[] points = FullHandCommonSchema.CopyPoints();
            Assert.AreEqual(Points, points.Length);

            var poseIds = new[] { 13, 14, 15, 16 };
            var poseNames = new[] { "left_elbow", "right_elbow", "left_wrist", "right_wrist" };
            for (int index = 0; index < 4; index++)
            {
                Assert.AreEqual(FullHandComponent.Pose, points[index].Component);
                Assert.AreEqual(poseIds[index], points[index].SourceIndex);
                Assert.AreEqual(poseNames[index], points[index].Name);
            }
            foreach (var block in new[]
                     { new object[] { LeftBase, FullHandComponent.LeftHand },
                       new object[] { RightBase, FullHandComponent.RightHand } })
            {
                int at = (int)block[0];
                var component = (FullHandComponent)block[1];
                var names = new HashSet<string>();
                for (int joint = 0; joint < Hand; joint++)
                {
                    FullHandPointSpec point = points[at + joint];
                    Assert.AreEqual(component, point.Component);
                    Assert.AreEqual(joint, point.SourceIndex);
                    Assert.AreEqual(at + joint, point.Index);
                    names.Add(point.Name);
                }
                Assert.AreEqual(Hand, names.Count);
                // the thumb is FIRST, matching the numeric joint ids
                Assert.AreEqual("wrist", points[at].Name);
                Assert.AreEqual("thumb_cmc", points[at + 1].Name);
                Assert.AreEqual("thumb_tip", points[at + 4].Name);
                Assert.AreEqual("index_mcp", points[at + 5].Name);
                Assert.AreEqual("index_tip", points[at + 8].Name);
                Assert.AreEqual("pinky_tip", points[at + 20].Name);
            }
            Assert.AreEqual(new[] { 0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 25, 29, 33, 37, 41, 45 },
                            FullHandCommonSchema.CopyLegacy16PointIndices());
        }

        [Test]
        public void EveryHandJointIsRetainedAndDistinguishable()
        {
            FullHandCommonCapture capture = Adapted(Capture(3));
            var seen = new HashSet<string>();
            for (int joint = 0; joint < Hand; joint++)
            {
                Assert.IsTrue(capture.IsValid(0, LeftBase + joint), "joint " + joint);
                seen.Add(capture.Coordinate(0, LeftBase + joint, 0).ToString("R") + "|" +
                         capture.Coordinate(0, LeftBase + joint, 1).ToString("R"));
            }
            Assert.AreEqual(Hand, seen.Count, "every joint must be separately observable");
            Assert.AreEqual(Points * 3, capture.CopyCoordinates().Length / capture.FrameCount);
        }

        [Test]
        public void IntermediateJointChangeMovesOnlyThatPointAndLeavesLegacy16Unchanged()
        {
            const int joint = 6;                    // index_pip, absent from the frozen 16
            var moved = new Dictionary<int, double[]> { { joint, new[] { 0.44, 0.41 } } };
            FullHandCommonCapture before = Adapted(Capture(3));
            FullHandCommonCapture after = Adapted(Capture(3,
                left: frame => HandSet(frame, 0.30, 0.50, overrides: moved)));

            int changed = LeftBase + joint;
            Assert.AreNotEqual(before.Coordinate(0, changed, 0),
                               after.Coordinate(0, changed, 0));
            for (int frame = 0; frame < before.FrameCount; frame++)
                for (int point = 0; point < Points; point++)
                {
                    if (point == changed) continue;
                    Assert.AreEqual(before.IsValid(frame, point), after.IsValid(frame, point));
                    if (!before.IsValid(frame, point)) continue;
                    for (int axis = 0; axis < 3; axis++)
                        Assert.AreEqual(before.Coordinate(frame, point, axis),
                                        after.Coordinate(frame, point, axis), 0.0,
                                        "point " + point + " must not move");
                }
        }

        // ==== agreement with the frozen 16-point path ===========================

        [Test]
        public void Legacy16ProjectionMatchesAdaptCommon2DWhenEveryLandmarkIsValid()
        {
            AssertLegacyProjectionMatches(Capture(4));
        }

        [Test]
        public void Legacy16ProjectionMatchesAdaptCommon2DAcrossAnApprovedOneFrameGap()
        {
            const int joint = 8;                    // index_tip, kept by the frozen 16
            List<RawMediaPipeObservation> capture = Capture(3,
                left: frame => HandSet(frame, 0.30, 0.50, missingJoint: frame == 1 ? joint : -1));

            FullHandCommonCapture full = Adapted(capture);
            int point = LeftBase + joint;
            Assert.IsFalse(full.WasOriginallyValid(1, point));
            Assert.IsTrue(full.WasInterpolated(1, point), "the approved max_gap=1 fill applies");
            Assert.IsTrue(full.IsValid(1, point));
            AssertLegacyProjectionMatches(capture);
        }

        [Test]
        public void StrongDepthBoundaryPreventsTheGapFill()
        {
            const int joint = 8;
            List<RawMediaPipeObservation> capture = Capture(3,
                left: frame => HandSet(frame, 0.30, 0.50,
                                       missingJoint: frame == 1 ? joint : -1,
                                       strongAtMissing: true));

            FullHandCommonCapture full = Adapted(capture);
            int point = LeftBase + joint;
            Assert.IsFalse(full.WasInterpolated(1, point), "a strong boundary blocks the fill");
            Assert.IsFalse(full.IsValid(1, point));
            for (int axis = 0; axis < 3; axis++)
                Assert.IsNaN(full.Coordinate(1, point, axis), "a masked point stays NaN");
            AssertLegacyProjectionMatches(capture);
        }

        // ==== geometry ==========================================================

        [Test]
        public void NonSquareGeometryRestoresThePixelScaleWithDirectlyComputedCoordinates()
        {
            // Anchors at frame 0: shoulders (0.35,0.35) and (0.65,0.35) in a 1280x720
            // plane give origin (640,252) and shoulder width 384; hips below give
            // up = (0,-1,0). So x' = (nx*W - 640)/384 and y' = -(ny*H - 252)/384.
            const double originX = 640.0, originY = 252.0, scale = 384.0;
            var placed = new Dictionary<int, double[]>
            {
                { 5, new[] { 0.50, 0.50 } },
                { 7, new[] { 0.75, 0.25 } },
            };
            FullHandCommonCapture capture = Adapted(Capture(2,
                left: frame => HandSet(frame, 0.30, 0.50, overrides: placed)));

            foreach (KeyValuePair<int, double[]> pair in placed)
            {
                double expectedX = (pair.Value[0] * Width - originX) / scale;
                double expectedY = -(pair.Value[1] * Height - originY) / scale;
                int point = LeftBase + pair.Key;
                Assert.AreEqual(expectedX, capture.Coordinate(0, point, 0), 1e-12);
                Assert.AreEqual(expectedY, capture.Coordinate(0, point, 1), 1e-12);
                Assert.AreEqual(0.0, capture.Coordinate(0, point, 2), 0.0,
                                "channel 2 is structurally zero");
            }
            Assert.AreEqual(Width, capture.ImageWidth);
            Assert.AreEqual(Height, capture.ImageHeight);
        }

        [Test]
        public void MissingOrDriftingGeometryIsRefused()
        {
            Refused(Capture(3, geometry: frame => null), "geometry");
            var narrow = new Geo { W = 640, H = 480, TexW = 640, TexH = 480,
                                   CamW = 640, CamH = 480 };
            Refused(Capture(3, geometry: frame => frame == 1
                                ? Merged(narrow, narrow) : Standard()),
                    "geometry changed within the capture");
        }

        // ==== refusals ==========================================================

        [Test]
        public void MetricThreeDConventionIsRefusedRatherThanTreatedAsPixels()
        {
            Refused(Capture(3, convention: RawObservationConvention.ApprovedStep7B),
                    "MediaPipeNormalizedImage");
        }

        [Test]
        public void NonIncreasingTimestampsAreRefused()
        {
            // a non-finite timestamp cannot be built through the public observation
            // constructor, so only the ordering rule is exercised here
            Refused(Capture(3, timestamp: frame => frame == 2 ? 0.05 : 0.05 * (frame + 1)),
                    "strictly increasing");
        }

        [Test]
        public void EmptyCaptureAndFullyAbsentLandmarksAreRefused()
        {
            Refused(null, "at least one observation");
            Refused(new List<RawMediaPipeObservation>(), "at least one observation");

            var absent = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < 3; frame++)
                absent.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    RawLandmarkObservationSet.Absent(Pose),
                    RawLandmarkObservationSet.Absent(Hand),
                    RawLandmarkObservationSet.Absent(Hand),
                    Standard()));
            Refused(absent, "no valid normalized point");
        }

        // ==== copies ============================================================

        [Test]
        public void CopiesAreDefensiveAndTheInputObservationsAreUnchanged()
        {
            List<RawMediaPipeObservation> capture = Capture(3);
            double[] sourceBefore = capture[1].LeftHand.CopyCoordinates();
            bool[] sourceMaskBefore = capture[1].LeftHand.CopyValidMask();

            FullHandCommonCapture full = Adapted(capture);
            double[] coordinates = full.CopyCoordinates();
            bool[] mask = full.CopyValidMask();
            bool[] original = full.CopyOriginalValidMask();
            bool[] interpolated = full.CopyInterpolatedMask();
            coordinates[0] = 12345.0;
            mask[0] = !mask[0];
            original[0] = !original[0];
            interpolated[0] = !interpolated[0];

            Assert.AreNotEqual(12345.0, full.CopyCoordinates()[0]);
            Assert.AreEqual(full.IsValid(0, 0), full.CopyValidMask()[0]);
            Assert.AreEqual(full.WasOriginallyValid(0, 0), full.CopyOriginalValidMask()[0]);
            Assert.AreEqual(full.WasInterpolated(0, 0), full.CopyInterpolatedMask()[0]);

            int[] projection = FullHandCommonSchema.CopyLegacy16PointIndices();
            projection[0] = 45;
            Assert.AreEqual(0, FullHandCommonSchema.CopyLegacy16PointIndices()[0]);

            Assert.AreEqual(sourceBefore, capture[1].LeftHand.CopyCoordinates());
            Assert.AreEqual(sourceMaskBefore, capture[1].LeftHand.CopyValidMask());
        }
    }
}
