// F51 edit-mode tests: row order, guards, point mapping and convention selection only.
// No scene, camera, detector, model payload or private recording is involved, and nothing
// here asserts any recognition accuracy or performance claim.
using System;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Core.Tests
{
    [TestFixture]
    public sealed class UprightInferencePixelBridgeTests
    {
        static Color32[] Ramp(int width, int height)
        {
            var pixels = new Color32[width * height];
            for (int index = 0; index < pixels.Length; index++)
                pixels[index] = new Color32((byte)index, 0, 0, 255);
            return pixels;
        }

        static Step9DInferenceFrameGeometry Geometry(bool upright)
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                null, null, null, 640, 480, 0, false, 640, 480, 640, 480, "SRGBA",
                Step9DPreInferenceTransformState.None,
                upright ? Step9DPreInferenceTransformState.Applied
                        : Step9DPreInferenceTransformState.None,
                null,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, 0L, 0,
                upright ? UprightInferencePixelBridge.RowOrderTopDown
                        : UprightInferencePixelBridge.RowOrderBottomUp);
        }

        static Step9DMergedInferenceGeometry Merged(bool handUpright, bool poseUpright)
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(Geometry(handUpright)),
                Step9DGeometryAcquisitionResult.ForAvailable(Geometry(poseUpright)));
        }

        [Test]
        public void RowsAreReversedForEvenAndOddHeights()
        {
            var source = Ramp(2, 3);                       // rows {0,1} {2,3} {4,5}
            var destination = new Color32[6];
            UprightInferencePixelBridge.ReversePixelRows(source, 2, 3, destination);
            var expected = new byte[] { 4, 5, 2, 3, 0, 1 };
            for (int index = 0; index < expected.Length; index++)
                Assert.AreEqual(expected[index], destination[index].r, "index " + index);

            var wide = Ramp(3, 2);                         // non-square, even height
            var flipped = new Color32[6];
            UprightInferencePixelBridge.ReversePixelRows(wide, 3, 2, flipped);
            Assert.AreEqual(3, flipped[0].r);
            Assert.AreEqual(0, flipped[3].r);

            var round = new Color32[6];
            UprightInferencePixelBridge.ReversePixelRows(destination, 2, 3, round);
            for (int index = 0; index < round.Length; index++)
                Assert.AreEqual(source[index].r, round[index].r, "reversal is an involution");
        }

        [Test]
        public void MalformedSizesAliasingAndOverflowAreRejected()
        {
            var source = Ramp(2, 2);
            Assert.Throws<ArgumentNullException>(
                () => UprightInferencePixelBridge.ReversePixelRows(null, 2, 2, new Color32[4]));
            Assert.Throws<ArgumentNullException>(
                () => UprightInferencePixelBridge.ReversePixelRows(source, 2, 2, null));
            Assert.Throws<ArgumentException>(
                () => UprightInferencePixelBridge.ReversePixelRows(source, 2, 2, source));
            Assert.Throws<ArgumentException>(
                () => UprightInferencePixelBridge.ReversePixelRows(source, 2, 3, new Color32[6]));
            Assert.Throws<ArgumentException>(
                () => UprightInferencePixelBridge.ReversePixelRows(source, 2, 2, new Color32[3]));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => UprightInferencePixelBridge.ReversePixelRows(source, 0, 2, new Color32[4]));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => UprightInferencePixelBridge.ReversePixelRows(source, 65536, 65536,
                                                                   new Color32[4]));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => UprightInferencePixelBridge.EnsureBuffer(null, 0));
            var buffer = UprightInferencePixelBridge.EnsureBuffer(null, 4);
            Assert.AreSame(buffer, UprightInferencePixelBridge.EnsureBuffer(buffer, 4),
                           "an exact-length buffer is reused, not reallocated per frame");
            Assert.AreNotSame(buffer, UprightInferencePixelBridge.EnsureBuffer(buffer, 5));
        }

        [Test]
        public void ThePointMapIsAnInvolutionAndKeepsNonFiniteValues()
        {
            var point = new Vector3(0.25f, 0.75f, -0.125f);
            Vector3 mapped = UprightInferencePixelBridge.ToSourceTexturePoint(point);
            Assert.AreEqual(0.25f, mapped.x, 1e-6f);
            Assert.AreEqual(0.25f, mapped.y, 1e-6f);
            Assert.AreEqual(-0.125f, mapped.z, 1e-6f, "z is untouched");
            Vector3 back = UprightInferencePixelBridge.ToSourceTexturePoint(mapped);
            Assert.AreEqual(point.y, back.y, 1e-6f);

            Vector3 nan = UprightInferencePixelBridge.ToSourceTexturePoint(
                new Vector3(float.NaN, float.NaN, float.NaN));
            Assert.IsTrue(float.IsNaN(nan.x) && float.IsNaN(nan.y) && float.IsNaN(nan.z),
                          "a missing point is never fabricated into a finite one");
            Vector3 infinite = UprightInferencePixelBridge.ToSourceTexturePoint(
                new Vector3(1f, float.PositiveInfinity, 1f));
            Assert.IsTrue(float.IsInfinity(infinite.y));
        }

        [Test]
        public void OnlyTheVerifiedDesktopCameraGeometryIsAccepted()
        {
            Assert.IsTrue(UprightInferencePixelBridge.IsSupportedCameraGeometry(0, false));
            Assert.IsTrue(UprightInferencePixelBridge.IsSupportedCameraGeometry(360, false));
            Assert.IsTrue(UprightInferencePixelBridge.IsSupportedCameraGeometry(-360, false));
            Assert.IsFalse(UprightInferencePixelBridge.IsSupportedCameraGeometry(0, true));
            Assert.IsFalse(UprightInferencePixelBridge.IsSupportedCameraGeometry(90, false));
            Assert.IsFalse(UprightInferencePixelBridge.IsSupportedCameraGeometry(180, false));
            Assert.IsFalse(UprightInferencePixelBridge.IsSupportedCameraGeometry(270, true));
            StringAssert.Contains(UprightInferencePixelBridge.UnsupportedGeometryCode,
                UprightInferencePixelBridge.DescribeUnsupportedGeometry(90, true));
            Assert.AreEqual("WIS_UPRIGHT_INFERENCE",
                            UprightInferencePixelBridge.EnvironmentFlagName);
        }

        [Test]
        public void TheUprightConventionIsSelectedOnlyWhenBothProvidersAgree()
        {
            Assert.IsTrue(UprightInferencePixelBridge.IsUprightTextureGeometry(Merged(true, true)));
            Assert.IsFalse(UprightInferencePixelBridge.IsUprightTextureGeometry(Merged(false, false)));
            Assert.IsFalse(UprightInferencePixelBridge.IsUprightTextureGeometry(Merged(true, false)),
                           "a mixed pair never manufactures agreement");
            Assert.IsFalse(UprightInferencePixelBridge.IsUprightTextureGeometry(Merged(false, true)));
            Assert.IsFalse(UprightInferencePixelBridge.IsUprightTextureGeometry(null),
                           "a test frame without geometry keeps the legacy convention");
            Assert.IsFalse(UprightInferencePixelBridge.IsUprightTextureGeometry(
                new Step9DMergedInferenceGeometry(
                    Step9DGeometryAcquisitionResult.ForAvailable(Geometry(true)), null)));

            // the shipped convention constant is unchanged, and the new one is distinct
            Assert.AreEqual("mediapipe_tasks_webcam_normalized_image_v0_16_3",
                            RawObservationConvention.LiveMediaPipe.Id);
            Assert.AreEqual("MediaPipe NormalizedLandmark x-right/y-down/z-source-relative; "
                            + "no axis rewrite", RawObservationConvention.LiveMediaPipe.AxisAndDepth);
            Assert.AreNotEqual(RawObservationConvention.LiveMediaPipe.Id,
                               RawObservationConvention.LiveMediaPipeUprightTexture.Id);
            Assert.AreEqual(RawObservationCoordinateSpace.MediaPipeNormalizedImage,
                            RawObservationConvention.LiveMediaPipeUprightTexture.CoordinateSpace);
        }

        [Test]
        public void MappedPointsKeepTheSegmenterBottomCompatibleOrdering()
        {
            // In an upright top-down image a hanging wrist sits BELOW the elbow, so its image
            // y is larger. After the map the legacy rule wrist.y <= elbow.y still holds.
            Vector3 elbow = UprightInferencePixelBridge.ToSourceTexturePoint(
                new Vector3(0.40f, 0.50f, 0f));
            Vector3 wrist = UprightInferencePixelBridge.ToSourceTexturePoint(
                new Vector3(0.41f, 0.70f, 0f));
            Assert.LessOrEqual(wrist.y, elbow.y);
            Assert.AreEqual(0.20f, elbow.y - wrist.y, 1e-6f,
                            "the vertical separation is preserved in magnitude");
            Vector3 raised = UprightInferencePixelBridge.ToSourceTexturePoint(
                new Vector3(0.41f, 0.30f, 0f));
            Assert.Greater(raised.y, elbow.y, "a raised wrist is still not bottom compatible");
        }

        [Test]
        public void EveryPoseAndHandPointIsMappedIdenticallyWithoutReordering()
        {
            var pose = new Vector3[33];
            for (int index = 0; index < pose.Length; index++)
                pose[index] = new Vector3(0.01f * index, 0.02f * index, 0.03f * index);
            var left = new Vector3[21];
            var right = new Vector3[21];
            for (int index = 0; index < 21; index++)
            {
                left[index] = new Vector3(0.1f + 0.01f * index, 0.2f + 0.01f * index, 0.5f);
                right[index] = new Vector3(0.6f + 0.01f * index, 0.3f + 0.01f * index, -0.5f);
            }
            foreach (var set in new[] { pose, left, right })
                for (int index = 0; index < set.Length; index++)
                {
                    Vector3 mapped = UprightInferencePixelBridge.ToSourceTexturePoint(set[index]);
                    Assert.AreEqual(set[index].x, mapped.x, 1e-6f, "x is untouched");
                    Assert.AreEqual(1f - set[index].y, mapped.y, 1e-6f);
                    Assert.AreEqual(set[index].z, mapped.z, 1e-6f);
                }
            // ordering and per-hand identity are positional only: nothing in this bridge can
            // exchange the left and right arrays or their handedness labels
            Assert.AreEqual(33, pose.Length);
            Assert.AreEqual(21, left.Length);
            Assert.AreEqual(21, right.Length);
        }
    }
}
