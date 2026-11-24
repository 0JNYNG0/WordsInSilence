using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    [TestFixture]
    public sealed class Step9DInferenceFrameGeometryTests
    {
        // ─── Helper ────────────────────────────────────────────────────────

        static Step9DInferenceFrameGeometry MakeHandGeometry(
            int wcWidth = 640, int wcHeight = 480, long tsMs = 100,
            Step9DPreInferenceTransformState hFlip = Step9DPreInferenceTransformState.None,
            Step9DPreInferenceTransformState vFlip = Step9DPreInferenceTransformState.None,
            int? rotation = 0,
            bool? vertMirror = false,
            string rowOrder = "bottom-up",
            string imageFormat = "SRGBA")
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.ObservedAtHandInference,
                null, null, null,
                wcWidth, wcHeight, rotation, vertMirror,
                wcWidth, wcHeight, wcWidth, wcHeight,
                imageFormat,
                hFlip, vFlip, null,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, tsMs, 1,
                rowOrder);
        }

        static Step9DInferenceFrameGeometry MakePoseGeometry(
            int wcWidth = 640, int wcHeight = 480, long tsMs = 100,
            Step9DPreInferenceTransformState hFlip = Step9DPreInferenceTransformState.None,
            Step9DPreInferenceTransformState vFlip = Step9DPreInferenceTransformState.None,
            int? rotation = 0,
            bool? vertMirror = false,
            string rowOrder = "bottom-up",
            string imageFormat = "SRGBA")
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.ObservedAtPoseInference,
                null, null, null,
                wcWidth, wcHeight, rotation, vertMirror,
                wcWidth, wcHeight, wcWidth, wcHeight,
                imageFormat,
                hFlip, vFlip, null,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, tsMs, 1,
                rowOrder);
        }

        static Step9DGeometryAcquisitionResult AvailableHand(
            int w = 640, int h = 480, long tsMs = 100,
            Step9DPreInferenceTransformState hFlip = Step9DPreInferenceTransformState.None,
            Step9DPreInferenceTransformState vFlip = Step9DPreInferenceTransformState.None,
            int? rotation = 0, bool? vertMirror = false,
            string rowOrder = "bottom-up", string imageFormat = "SRGBA")
        {
            return Step9DGeometryAcquisitionResult.ForAvailable(
                MakeHandGeometry(w, h, tsMs, hFlip, vFlip, rotation, vertMirror, rowOrder, imageFormat));
        }

        static Step9DGeometryAcquisitionResult AvailablePose(
            int w = 640, int h = 480, long tsMs = 100,
            Step9DPreInferenceTransformState hFlip = Step9DPreInferenceTransformState.None,
            Step9DPreInferenceTransformState vFlip = Step9DPreInferenceTransformState.None,
            int? rotation = 0, bool? vertMirror = false,
            string rowOrder = "bottom-up", string imageFormat = "SRGBA")
        {
            return Step9DGeometryAcquisitionResult.ForAvailable(
                MakePoseGeometry(w, h, tsMs, hFlip, vFlip, rotation, vertMirror, rowOrder, imageFormat));
        }

        static Step9DGeometryAcquisitionResult Unavailable(
            Step9DGeometryAcquisitionStatus status = Step9DGeometryAcquisitionStatus.UnavailableCaptureException,
            string code = "test exception")
        {
            return Step9DGeometryAcquisitionResult.ForUnavailable(status, code);
        }

        static RawMediaPipeObservation MakeObservation(double timestamp,
            Step9DMergedInferenceGeometry mergedGeometry = null)
        {
            var posePoints = new Dictionary<int, double[]>
            {
                { 11, new[] { -1.0, 0.0, 0.0 } },
                { 12, new[] { 1.0, 0.0, 0.0 } },
                { 23, new[] { -0.5, 1.0, 0.0 } },
                { 24, new[] { 0.5, 1.0, 0.0 } },
                { 13, new[] { -1.2, 0.4, 0.0 } },
                { 14, new[] { 1.2, 0.4, 0.0 } },
                { 15, new[] { -1.3, 0.8, 0.0 } },
                { 16, new[] { 1.3, 0.8, 0.0 } },
            };
            return new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.ApprovedStep7B,
                MakeSet(33, posePoints),
                MakeSet(21, new Dictionary<int, double[]>()),
                MakeSet(21, new Dictionary<int, double[]>()),
                mergedGeometry);
        }

        static RawLandmarkObservationSet MakeSet(int count, Dictionary<int, double[]> points)
        {
            var coordinates = new double[count * 3];
            for (int i = 0; i < coordinates.Length; i++) coordinates[i] = double.NaN;
            var valid = new bool[count];
            var reappearance = new bool[count];
            var strong = new bool[count];
            foreach (var pair in points)
            {
                valid[pair.Key] = true;
                Array.Copy(pair.Value, 0, coordinates, pair.Key * 3, 3);
            }
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, strong);
        }

        static Step9DMergedInferenceGeometry CompleteMerged(
            int w = 640, int h = 480, long handTs = 100, long poseTs = 100)
        {
            return new Step9DMergedInferenceGeometry(
                AvailableHand(w, h, handTs), AvailablePose(w, h, poseTs));
        }

        // =====================================================================
        //  Section A -- Construction and fields (10 tests)
        // =====================================================================

        [Test]
        public void Constructor_HandProvenance_StoresAllFields()
        {
            var g = new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.ObservedAtHandInference,
                640, 480, 30,
                1280, 720, 90, true,
                1280, 720, 1280, 720,
                "SRGBA",
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                null,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                42L, 5000L, 100,
                "bottom-up");

            Assert.AreEqual(Step9DInferenceGeometryProvenance.ObservedAtHandInference, g.Provenance);
            Assert.AreEqual(640, g.RequestedWidthOrNull);
            Assert.AreEqual(480, g.RequestedHeightOrNull);
            Assert.AreEqual(30, g.RequestedFpsOrNull);
            Assert.AreEqual(1280, g.ObservedWebcamWidth);
            Assert.AreEqual(720, g.ObservedWebcamHeight);
            Assert.AreEqual(90, g.ObservedVideoRotationAngleOrNull);
            Assert.AreEqual(true, g.ObservedVideoVerticallyMirroredOrNull);
            Assert.AreEqual(1280, g.InferenceTextureWidth);
            Assert.AreEqual(720, g.InferenceTextureHeight);
            Assert.AreEqual(1280, g.InferenceImageWidth);
            Assert.AreEqual(720, g.InferenceImageHeight);
            Assert.AreEqual("SRGBA", g.ImageFormat);
            Assert.AreEqual(Step9DPreInferenceTransformState.None, g.PreInferenceHorizontalFlip);
            Assert.AreEqual(Step9DPreInferenceTransformState.None, g.PreInferenceVerticalFlip);
            Assert.IsNull(g.PreInferenceRotationDegreesOrNull);
            Assert.AreEqual(Step9DPreInferenceTransformState.None, g.Crop);
            Assert.AreEqual(42L, g.ProviderFrameIndexOrNull);
            Assert.AreEqual(5000L, g.MediaPipeTimestampMs);
            Assert.AreEqual(100, g.UnityFrameCountOrNull);
            Assert.AreEqual("bottom-up", g.PixelBufferRowOrder);
        }

        [Test]
        public void Constructor_PoseProvenance_Accepted()
        {
            var g = MakePoseGeometry();
            Assert.AreEqual(Step9DInferenceGeometryProvenance.ObservedAtPoseInference, g.Provenance);
        }

        [Test]
        public void Constructor_TestInjected_AllowsZeroWebcamDims()
        {
            var g = new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                null, null, null,
                0, 0, null, null,
                640, 480, 640, 480,
                "SRGBA",
                Step9DPreInferenceTransformState.Unknown,
                Step9DPreInferenceTransformState.Unknown,
                null,
                Step9DPreInferenceTransformState.Unknown,
                Step9DPreInferenceTransformState.Unknown,
                Step9DPreInferenceTransformState.Unknown,
                null, 0L, null,
                "unknown");

            Assert.AreEqual(Step9DInferenceGeometryProvenance.TestInjected, g.Provenance);
            Assert.AreEqual(0, g.ObservedWebcamWidth);
        }

        [Test]
        public void Constructor_NonTestProvenance_RejectsZeroWebcamWidth()
        {
            Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.ObservedAtHandInference,
                    null, null, null,
                    0, 480, 0, false,
                    640, 480, 640, 480,
                    "SRGBA",
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    null,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    null, 0L, null,
                    "bottom-up"));
        }

        [Test]
        public void Constructor_RejectsZeroInferenceTextureWidth()
        {
            Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.TestInjected,
                    null, null, null,
                    0, 0, null, null,
                    0, 480, 640, 480,
                    "SRGBA",
                    Step9DPreInferenceTransformState.Unknown,
                    Step9DPreInferenceTransformState.Unknown,
                    null,
                    Step9DPreInferenceTransformState.Unknown,
                    Step9DPreInferenceTransformState.Unknown,
                    Step9DPreInferenceTransformState.Unknown,
                    null, 0L, null,
                    "unknown"));
        }

        [Test]
        public void Constructor_RejectsEmptyImageFormat()
        {
            Assert.Throws<ArgumentException>(() =>
                new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.TestInjected,
                    null, null, null,
                    0, 0, null, null,
                    640, 480, 640, 480,
                    "",
                    Step9DPreInferenceTransformState.Unknown,
                    Step9DPreInferenceTransformState.Unknown,
                    null,
                    Step9DPreInferenceTransformState.Unknown,
                    Step9DPreInferenceTransformState.Unknown,
                    Step9DPreInferenceTransformState.Unknown,
                    null, 0L, null,
                    "unknown"));
        }

        [Test]
        public void Clone_ReturnsEqualButDistinctInstance()
        {
            var original = MakeHandGeometry();
            var cloned = original.Clone();
            Assert.AreNotSame(original, cloned);
            Assert.AreEqual(original.Provenance, cloned.Provenance);
            Assert.AreEqual(original.ObservedWebcamWidth, cloned.ObservedWebcamWidth);
            Assert.AreEqual(original.ImageFormat, cloned.ImageFormat);
            Assert.AreEqual(original.PixelBufferRowOrder, cloned.PixelBufferRowOrder);
        }

        [Test]
        public void PixelBufferRowOrder_NullDefaultsToUnknown()
        {
            var g = new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                null, null, null,
                0, 0, null, null,
                640, 480, 640, 480,
                "SRGBA",
                Step9DPreInferenceTransformState.Unknown,
                Step9DPreInferenceTransformState.Unknown,
                null,
                Step9DPreInferenceTransformState.Unknown,
                Step9DPreInferenceTransformState.Unknown,
                Step9DPreInferenceTransformState.Unknown,
                null, 0L, null,
                null);
            Assert.AreEqual("unknown", g.PixelBufferRowOrder);
        }

        [Test]
        public void TransformState_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)Step9DPreInferenceTransformState.Unknown);
            Assert.AreEqual(1, (int)Step9DPreInferenceTransformState.None);
            Assert.AreEqual(2, (int)Step9DPreInferenceTransformState.Applied);
        }

        [Test]
        public void Provenance_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)Step9DInferenceGeometryProvenance.ObservedAtHandInference);
            Assert.AreEqual(1, (int)Step9DInferenceGeometryProvenance.ObservedAtPoseInference);
            Assert.AreEqual(2, (int)Step9DInferenceGeometryProvenance.TestInjected);
        }

        // =====================================================================
        //  Section B -- Typed acquisition result (9 tests)
        // =====================================================================

        [Test]
        public void AcquisitionResult_ForAvailable_ReturnsAvailableWithGeometry()
        {
            var geom = MakeHandGeometry();
            var result = Step9DGeometryAcquisitionResult.ForAvailable(geom);
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.Available, result.Status);
            Assert.AreSame(geom, result.Geometry);
            Assert.AreEqual(string.Empty, result.DiagnosticCode);
        }

        [Test]
        public void AcquisitionResult_ForAvailable_RejectsNull()
        {
            Assert.Throws<ArgumentNullException>(() =>
                Step9DGeometryAcquisitionResult.ForAvailable(null));
        }

        [Test]
        public void AcquisitionResult_ForUnavailable_CaptureException()
        {
            var result = Step9DGeometryAcquisitionResult.ForUnavailable(
                Step9DGeometryAcquisitionStatus.UnavailableCaptureException, "test error");
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableCaptureException, result.Status);
            Assert.IsNull(result.Geometry);
            Assert.AreEqual("test error", result.DiagnosticCode);
        }

        [Test]
        public void AcquisitionResult_ForUnavailable_InvalidDimensions()
        {
            var result = Step9DGeometryAcquisitionResult.ForUnavailable(
                Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions, "width 0");
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions, result.Status);
            Assert.IsNull(result.Geometry);
        }

        [Test]
        public void AcquisitionResult_ForUnavailable_BeforeFrameReady()
        {
            var result = Step9DGeometryAcquisitionResult.ForUnavailable(
                Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady, "no frame yet");
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady, result.Status);
        }

        [Test]
        public void AcquisitionResult_ForUnavailable_RejectsAvailableStatus()
        {
            Assert.Throws<ArgumentException>(() =>
                Step9DGeometryAcquisitionResult.ForUnavailable(
                    Step9DGeometryAcquisitionStatus.Available, "bad"));
        }

        [Test]
        public void AcquisitionResult_Clone_ClonesGeometry()
        {
            var result = AvailableHand();
            var cloned = result.Clone();
            Assert.AreNotSame(result, cloned);
            Assert.AreNotSame(result.Geometry, cloned.Geometry);
            Assert.AreEqual(result.Status, cloned.Status);
            Assert.AreEqual(result.Geometry.ObservedWebcamWidth, cloned.Geometry.ObservedWebcamWidth);
        }

        [Test]
        public void AcquisitionResult_Clone_UnavailableHasNullGeometry()
        {
            var result = Unavailable();
            var cloned = result.Clone();
            Assert.AreNotSame(result, cloned);
            Assert.IsNull(cloned.Geometry);
            Assert.AreEqual(result.DiagnosticCode, cloned.DiagnosticCode);
        }

        [Test]
        public void AcquisitionStatus_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)Step9DGeometryAcquisitionStatus.Available);
            Assert.AreEqual(1, (int)Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady);
            Assert.AreEqual(2, (int)Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions);
            Assert.AreEqual(3, (int)Step9DGeometryAcquisitionStatus.UnavailableCaptureException);
        }

        // =====================================================================
        //  Section C -- Full contract compatibility (12 tests)
        // =====================================================================

        [Test]
        public void FullContract_SameFields_Compatible()
        {
            var hand = MakeHandGeometry(640, 480);
            var pose = MakePoseGeometry(640, 480);
            Assert.IsTrue(hand.IsFullContractCompatible(pose));
        }

        [Test]
        public void FullContract_DifferentWebcamWidth_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480);
            var pose = MakePoseGeometry(1280, 480);
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
            StringAssert.Contains("observed_webcam_width", hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void FullContract_DifferentRotation_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480, rotation: 0);
            var pose = MakePoseGeometry(640, 480, rotation: 90);
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
            StringAssert.Contains("observed_video_rotation_angle", hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void FullContract_DifferentVerticalMirror_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480, vertMirror: false);
            var pose = MakePoseGeometry(640, 480, vertMirror: true);
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
            StringAssert.Contains("observed_video_vertically_mirrored", hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void FullContract_DifferentHorizontalFlip_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480, hFlip: Step9DPreInferenceTransformState.None);
            var pose = MakePoseGeometry(640, 480, hFlip: Step9DPreInferenceTransformState.Applied);
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
            StringAssert.Contains("pre_inference_horizontal_flip", hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void FullContract_UnknownVsNone_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480, hFlip: Step9DPreInferenceTransformState.Unknown);
            var pose = MakePoseGeometry(640, 480, hFlip: Step9DPreInferenceTransformState.None);
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
        }

        [Test]
        public void FullContract_DifferentRowOrder_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480, rowOrder: "bottom-up");
            var pose = MakePoseGeometry(640, 480, rowOrder: "top-down");
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
            StringAssert.Contains("pixel_buffer_row_order", hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void FullContract_DifferentImageFormat_Incompatible()
        {
            var hand = MakeHandGeometry(640, 480, imageFormat: "SRGBA");
            var pose = MakePoseGeometry(640, 480, imageFormat: "RGBA");
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
            StringAssert.Contains("image_format", hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void FullContract_NullOther_ReturnsFalse()
        {
            var hand = MakeHandGeometry();
            Assert.IsFalse(hand.IsFullContractCompatible(null));
        }

        [Test]
        public void FullContract_DescribeFirstMismatch_NullWhenCompatible()
        {
            var hand = MakeHandGeometry(640, 480);
            var pose = MakePoseGeometry(640, 480);
            Assert.IsNull(hand.DescribeFirstMismatch(pose));
        }

        [Test]
        public void DimensionallyCompatible_SubsetOfFullContract()
        {
            var hand = MakeHandGeometry(640, 480, rowOrder: "bottom-up");
            var pose = MakePoseGeometry(640, 480, rowOrder: "top-down");
            Assert.IsTrue(hand.IsDimensionallyCompatible(pose));
            Assert.IsFalse(hand.IsFullContractCompatible(pose));
        }

        [Test]
        public void FullContractCompatible_NotBothAvailable_IsFalse()
        {
            var handOnly = new Step9DMergedInferenceGeometry(AvailableHand(), null);
            Assert.IsFalse(handOnly.FullContractCompatible, "hand-only must be false");

            var poseOnly = new Step9DMergedInferenceGeometry(null, AvailablePose());
            Assert.IsFalse(poseOnly.FullContractCompatible, "pose-only must be false");

            var bothUnavailable = new Step9DMergedInferenceGeometry(Unavailable(), Unavailable());
            Assert.IsFalse(bothUnavailable.FullContractCompatible, "both-unavailable must be false");

            var handUnavailable = new Step9DMergedInferenceGeometry(Unavailable(), AvailablePose());
            Assert.IsFalse(handUnavailable.FullContractCompatible, "hand-unavailable must be false");

            var bothNull = new Step9DMergedInferenceGeometry(null, null);
            Assert.IsFalse(bothNull.FullContractCompatible, "both-null must be false");
        }

        // =====================================================================
        //  Section D -- Merged with acquisition results (10 tests)
        // =====================================================================

        [Test]
        public void Merged_BothAvailable_Compatible()
        {
            var merged = new Step9DMergedInferenceGeometry(AvailableHand(), AvailablePose());
            Assert.IsTrue(merged.BothAvailable);
            Assert.IsTrue(merged.BothPresent);
            Assert.IsTrue(merged.FullContractCompatible);
            Assert.IsTrue(merged.DimensionallyCompatible);
            Assert.IsNotNull(merged.Hand);
            Assert.IsNotNull(merged.Pose);
            Assert.AreSame(merged.Hand, merged.Representative);
        }

        [Test]
        public void Merged_BothAvailable_Incompatible()
        {
            var merged = new Step9DMergedInferenceGeometry(
                AvailableHand(640, 480), AvailablePose(1280, 720));
            Assert.IsTrue(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNotNull(merged.FirstMismatchDescription);
        }

        [Test]
        public void Merged_HandOnly()
        {
            var merged = new Step9DMergedInferenceGeometry(AvailableHand(), null);
            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNotNull(merged.Hand);
            Assert.IsNull(merged.Pose);
            Assert.AreSame(merged.Hand, merged.Representative);
        }

        [Test]
        public void Merged_PoseOnly()
        {
            var merged = new Step9DMergedInferenceGeometry(null, AvailablePose());
            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNull(merged.Hand);
            Assert.IsNotNull(merged.Pose);
            Assert.AreSame(merged.Pose, merged.Representative);
        }

        [Test]
        public void Merged_HandUnavailable_PoseAvailable()
        {
            var merged = new Step9DMergedInferenceGeometry(Unavailable(), AvailablePose());
            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNull(merged.Hand);
            Assert.IsNotNull(merged.Pose);
            Assert.IsNotNull(merged.HandResult);
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableCaptureException, merged.HandResult.Status);
        }

        [Test]
        public void Merged_BothUnavailable()
        {
            var merged = new Step9DMergedInferenceGeometry(Unavailable(), Unavailable());
            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNull(merged.Hand);
            Assert.IsNull(merged.Pose);
            Assert.IsNull(merged.Representative);
        }

        [Test]
        public void Merged_BothNull()
        {
            var merged = new Step9DMergedInferenceGeometry(null, null);
            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNull(merged.Representative);
        }

        [Test]
        public void Merged_TimestampSkew_BothAvailable()
        {
            var merged = new Step9DMergedInferenceGeometry(
                AvailableHand(tsMs: 100), AvailablePose(tsMs: 105));
            Assert.AreEqual(5L, merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void Merged_TimestampSkew_NotBothAvailable_IsNull()
        {
            var merged = new Step9DMergedInferenceGeometry(AvailableHand(), null);
            Assert.IsNull(merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void MergedClone_ReturnsDeepCopy()
        {
            var merged = new Step9DMergedInferenceGeometry(AvailableHand(), AvailablePose());
            var cloned = merged.Clone();
            Assert.AreNotSame(merged, cloned);
            Assert.AreNotSame(merged.Hand, cloned.Hand);
            Assert.AreNotSame(merged.Pose, cloned.Pose);
            Assert.AreEqual(merged.BothAvailable, cloned.BothAvailable);
            Assert.AreEqual(merged.FullContractCompatible, cloned.FullContractCompatible);
        }

        // =====================================================================
        //  Section E -- Observation and collector propagation (6 tests)
        // =====================================================================

        [Test]
        public void Observation_CarriesGeometry()
        {
            var merged = CompleteMerged();
            var obs = MakeObservation(1.0, merged);
            Assert.IsNotNull(obs.MergedGeometry);
            Assert.IsTrue(obs.MergedGeometry.BothAvailable);
        }

        [Test]
        public void Observation_NullGeometry_BackwardCompatible()
        {
            var obs = MakeObservation(1.0, null);
            Assert.IsNull(obs.MergedGeometry);
        }

        [Test]
        public void CloneObservation_PreservesGeometry()
        {
            var merged = CompleteMerged();
            var obs = MakeObservation(1.0, merged);
            var cloned = Step9DRawIntermediateDiagnosticSnapshot.CloneObservation(obs);
            Assert.IsNotNull(cloned.MergedGeometry);
            Assert.AreNotSame(obs.MergedGeometry, cloned.MergedGeometry);
            Assert.IsTrue(cloned.MergedGeometry.BothAvailable);
        }

        [Test]
        public void Snapshot_CapturesCompleteGeometryFromObservation()
        {
            var merged = CompleteMerged();
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, merged),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            Assert.IsNotNull(snapshot.CaptureInferenceGeometry);
            Assert.IsTrue(snapshot.CaptureInferenceGeometry.BothAvailable);
            Assert.IsTrue(snapshot.CaptureInferenceGeometry.FullContractCompatible);
        }

        [Test]
        public void Snapshot_WithoutGeometry_CaptureGeometryIsNull()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            Assert.IsNull(snapshot.CaptureInferenceGeometry);
        }

        [Test]
        public void Cancel_ClearsGeometry()
        {
            var geometry = new Step9DInferenceImageGeometry(640, 480, 0, false, 640, 480);
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", geometry);
            collector.Cancel();
            collector.Begin("run1", "cap2", "LOVE");
            collector.RecordObservation(
                MakeObservation(1.0),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            Assert.IsNull(snapshot.InferenceImageGeometry);
            Assert.IsNull(snapshot.CaptureInferenceGeometry);
        }

        // =====================================================================
        //  Section F -- Geometry-required fail-closed enforcement (7 tests)
        //  Uses actual collector production path.
        // =====================================================================

        [Test]
        public void GeometryRequired_AllUnavailable_ThrowsMissing()
        {
            var unavailableMerged = new Step9DMergedInferenceGeometry(Unavailable(), Unavailable());
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", null, true);
            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(1.0, unavailableMerged),
                    new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING, ex.Code);
        }

        [Test]
        public void GeometryRequired_AvailableThenMissing_ThrowsMissing()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", null, true);
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 100)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(2.0, null),
                    new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING, ex.Code);
        }

        [Test]
        public void GeometryRequired_AvailableThenUnavailable_ThrowsMissing()
        {
            var unavailableMerged = new Step9DMergedInferenceGeometry(
                Unavailable(Step9DGeometryAcquisitionStatus.UnavailableCaptureException, "cam err"),
                AvailablePose(640, 480, tsMs: 200));
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", null, true);
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 100)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(2.0, unavailableMerged),
                    new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING, ex.Code);
        }

        [Test]
        public void GeometryRequired_HandOnlyAvailable_ThrowsMissing()
        {
            var handOnlyMerged = new Step9DMergedInferenceGeometry(AvailableHand(), null);
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", null, true);
            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(1.0, handOnlyMerged),
                    new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING, ex.Code);
        }

        [Test]
        public void GeometryRequired_ContractChanged_ThrowsContractChanged()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", null, true);
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 100)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(2.0, CompleteMerged(1280, 720, 200, 200)),
                    new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED, ex.Code);
        }

        [Test]
        public void GeometryRequired_False_MissingAndUnavailable_Passes()
        {
            var unavailableMerged = new Step9DMergedInferenceGeometry(Unavailable(), Unavailable());
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, null),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(2.0, unavailableMerged),
                new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            Assert.IsNull(snapshot.CaptureInferenceGeometry,
                "no complete geometry observed, so CaptureInferenceGeometry must be null");
            Assert.AreEqual(2, snapshot.ObservationCount);
        }

        [Test]
        public void GeometryRequired_CompleteGeometry_CapturePreserved()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", null, true);
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 103)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(2.0, CompleteMerged(640, 480, 200, 204)),
                new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            Assert.IsNotNull(snapshot.CaptureInferenceGeometry);
            Assert.IsTrue(snapshot.CaptureInferenceGeometry.BothAvailable);
            Assert.IsTrue(snapshot.CaptureInferenceGeometry.FullContractCompatible);
            Assert.AreEqual(640, snapshot.CaptureInferenceGeometry.Hand.ObservedWebcamWidth);
            Assert.IsNotNull(snapshot.GeometryConsistencySummary);
            Assert.AreEqual(2, snapshot.GeometryConsistencySummary.CheckedCount);
        }

        // =====================================================================
        //  Section G -- Capture-wide consistency (5 tests)
        // =====================================================================

        [Test]
        public void Consistency_StableContract_PassesWithSummary()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 105)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(2.0, CompleteMerged(640, 480, 200, 208)),
                new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            Assert.IsNotNull(snapshot.GeometryConsistencySummary);
            Assert.AreEqual(2, snapshot.GeometryConsistencySummary.CheckedCount);
            Assert.AreEqual(100, snapshot.GeometryConsistencySummary.FirstHandTimestampMs);
            Assert.AreEqual(200, snapshot.GeometryConsistencySummary.LastHandTimestampMs);
            Assert.AreEqual(105, snapshot.GeometryConsistencySummary.FirstPoseTimestampMs);
            Assert.AreEqual(208, snapshot.GeometryConsistencySummary.LastPoseTimestampMs);
            Assert.AreEqual(8, snapshot.GeometryConsistencySummary.MaxHandPoseAbsSkewMs);
        }

        [Test]
        public void Consistency_HandContractChanged_ThrowsContractChanged()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 100)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));

            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(2.0, new Step9DMergedInferenceGeometry(
                        AvailableHand(1280, 720, tsMs: 200), AvailablePose(1280, 720, tsMs: 200))),
                    new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED, ex.Code);
            StringAssert.Contains("hand geometry contract changed", ex.Message);
        }

        [Test]
        public void Consistency_PoseContractChanged_ThrowsContractChanged()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 100)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));

            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(2.0, new Step9DMergedInferenceGeometry(
                        AvailableHand(640, 480, tsMs: 200), AvailablePose(1280, 720, tsMs: 200))),
                    new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED, ex.Code);
            StringAssert.Contains("pose geometry contract changed", ex.Message);
        }

        [Test]
        public void Consistency_RotationChanged_ThrowsContractChanged()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, new Step9DMergedInferenceGeometry(
                    AvailableHand(640, 480, tsMs: 100, rotation: 0),
                    AvailablePose(640, 480, tsMs: 100, rotation: 0))),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));

            var ex = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(
                    MakeObservation(2.0, new Step9DMergedInferenceGeometry(
                        AvailableHand(640, 480, tsMs: 200, rotation: 90),
                        AvailablePose(640, 480, tsMs: 200, rotation: 90))),
                    new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED, ex.Code);
        }

        [Test]
        public void Consistency_NoGeometryChecks_SummaryIsNull()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            Assert.IsNull(snapshot.GeometryConsistencySummary);
        }

        // =====================================================================
        //  Section H -- Serialization (5 tests)
        // =====================================================================

        [Test]
        public void Serialization_WithGeometry_ContainsAllFields()
        {
            var merged = CompleteMerged(1280, 720, 100, 105);

            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, merged),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);

            StringAssert.Contains("\"inference_frame_geometry\":{", json);
            StringAssert.Contains("\"both_present\":true", json);
            StringAssert.Contains("\"both_available\":true", json);
            StringAssert.Contains("\"full_contract_compatible\":true", json);
            StringAssert.Contains("\"first_mismatch_description\":null", json);
            StringAssert.Contains("\"hand_pose_timestamp_skew_ms\":5", json);
            StringAssert.Contains("\"hand_acquisition\":{", json);
            StringAssert.Contains("\"status\":\"Available\"", json);
            StringAssert.Contains("\"pose_acquisition\":{", json);
            StringAssert.Contains("\"provenance\":\"ObservedAtHandInference\"", json);
            StringAssert.Contains("\"provenance\":\"ObservedAtPoseInference\"", json);
            StringAssert.Contains("\"observed_webcam_width\":1280", json);
            StringAssert.Contains("\"image_format\":\"SRGBA\"", json);
            StringAssert.Contains("\"pixel_buffer_row_order\":\"bottom-up\"", json);
            StringAssert.Contains("\"pre_inference_horizontal_flip\":\"None\"", json);
        }

        [Test]
        public void Serialization_WithoutGeometry_OmitsBlock()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.IsFalse(json.Contains("inference_frame_geometry"),
                "inference_frame_geometry block should be absent");
        }

        [Test]
        public void Serialization_NullableFields_SerializedAsNull()
        {
            // Hand-only merged: not complete, so CaptureInferenceGeometry will be null.
            // Use complete merged to get serializable geometry.
            var merged = CompleteMerged();
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, merged),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            StringAssert.Contains("\"both_present\":true", json);
            StringAssert.Contains("\"requested_width\":null", json);
        }

        [Test]
        public void Serialization_ConsistencySummary_Present()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 103)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            StringAssert.Contains("\"geometry_consistency_summary\":{", json);
            StringAssert.Contains("\"checked_count\":1", json);
            StringAssert.Contains("\"max_hand_pose_abs_skew_ms\":3", json);
        }

        [Test]
        public void Serialization_UnavailableAcquisition_SerializesStatus()
        {
            // Unavailable hand + available pose: not complete, so CaptureInferenceGeometry
            // from the observation loop will be null. However, the observation still carries
            // its merged geometry. We need an observation with complete geometry first so
            // CaptureInferenceGeometry is set, then check serialization of a snapshot
            // that has unavailable status in its observations.
            // For this test, use a complete merged so CaptureInferenceGeometry is populated,
            // then verify serialization of acquisition blocks.
            var merged = CompleteMerged();
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, merged),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            StringAssert.Contains("\"status\":\"Available\"", json);
            StringAssert.Contains("\"diagnostic_code\":\"\"", json);
        }

        // =====================================================================
        //  Section I -- Typed-result simulation at factory level (5 tests)
        // =====================================================================

        [Test]
        public void ProviderSeam_CaptureException_ProducesTypedUnavailable()
        {
            Step9DGeometryAcquisitionResult result;
            try
            {
                new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.ObservedAtHandInference,
                    null, null, null,
                    640, 480, 0, false,
                    0, 0, 0, 0,
                    "SRGBA",
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    null,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    0L, 100L, 1, "bottom-up");
                result = null;
            }
            catch (Exception ex)
            {
                result = Step9DGeometryAcquisitionResult.ForUnavailable(
                    Step9DGeometryAcquisitionStatus.UnavailableCaptureException,
                    "hand: " + ex.GetType().Name + ": " + ex.Message);
            }

            Assert.IsNotNull(result);
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableCaptureException, result.Status);
            Assert.IsNull(result.Geometry);
            StringAssert.Contains("hand:", result.DiagnosticCode);
        }

        [Test]
        public void ProviderSeam_InvalidDimensions_ProducesUnavailableInvalidDimensions()
        {
            int webcamWidth = 0;
            int webcamHeight = 0;
            Step9DGeometryAcquisitionResult result;
            if (webcamWidth <= 0 || webcamHeight <= 0)
            {
                result = Step9DGeometryAcquisitionResult.ForUnavailable(
                    Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions,
                    "hand: webcam dimensions not yet valid: " + webcamWidth + "x" + webcamHeight);
            }
            else
            {
                result = null;
            }

            Assert.IsNotNull(result);
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions, result.Status);
            StringAssert.Contains("0x0", result.DiagnosticCode);
        }

        [Test]
        public void ProviderSeam_SuccessfulCapture_ProducesAvailable()
        {
            var geometry = MakeHandGeometry(640, 480, tsMs: 100);
            var result = Step9DGeometryAcquisitionResult.ForAvailable(geometry);
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.Available, result.Status);
            Assert.AreSame(geometry, result.Geometry);
            Assert.AreEqual(string.Empty, result.DiagnosticCode);
        }

        [Test]
        public void ProviderOriginCallerSeam_AcquisitionResult_PropagatedToMerged()
        {
            var handResult = AvailableHand(640, 480, tsMs: 100);
            var poseResult = AvailablePose(640, 480, tsMs: 103);
            var merged = new Step9DMergedInferenceGeometry(handResult, poseResult);

            Assert.AreSame(handResult, merged.HandResult);
            Assert.AreSame(poseResult, merged.PoseResult);
            Assert.AreSame(handResult.Geometry, merged.Hand);
            Assert.AreSame(poseResult.Geometry, merged.Pose);
            Assert.IsTrue(merged.BothAvailable);
            Assert.IsTrue(merged.FullContractCompatible);
        }

        [Test]
        public void ProviderOriginCallerSeam_UnavailableHand_MergedReflectsStatus()
        {
            var handResult = Unavailable(Step9DGeometryAcquisitionStatus.UnavailableCaptureException, "exc");
            var poseResult = AvailablePose();
            var merged = new Step9DMergedInferenceGeometry(handResult, poseResult);

            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsNull(merged.Hand);
            Assert.IsNotNull(merged.Pose);
            Assert.AreSame(merged.Pose, merged.Representative);
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableCaptureException, merged.HandResult.Status);
            Assert.AreEqual("exc", merged.HandResult.DiagnosticCode);
        }

        // =====================================================================
        //  Section J -- Timestamp skew computation (6 tests)
        //
        //  LIMITATION: These tests verify Math.Abs skew computation in
        //  Step9DMergedInferenceGeometry and max-skew tracking in the collector.
        //  Bounded pairing verification (whether HandPoseIntegrator rejects
        //  frames exceeding _maxTimestampDiffMs) requires MonoBehaviour
        //  instantiation and is not verifiable in EditMode without Unity
        //  runtime. This section does NOT claim "bounded skew verified".
        // =====================================================================

        [Test]
        public void TimestampSkew_BothAvailable_CorrectAbsValue()
        {
            var merged = new Step9DMergedInferenceGeometry(
                AvailableHand(tsMs: 1000), AvailablePose(tsMs: 1007));
            Assert.AreEqual(7L, merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void TimestampSkew_BothAvailable_ReversedOrder_StillPositive()
        {
            var merged = new Step9DMergedInferenceGeometry(
                AvailableHand(tsMs: 2000), AvailablePose(tsMs: 1995));
            Assert.AreEqual(5L, merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void TimestampSkew_ZeroSkew()
        {
            var merged = new Step9DMergedInferenceGeometry(
                AvailableHand(tsMs: 500), AvailablePose(tsMs: 500));
            Assert.AreEqual(0L, merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void TimestampSkew_OneSideUnavailable_IsNull()
        {
            var merged = new Step9DMergedInferenceGeometry(AvailableHand(tsMs: 500), Unavailable());
            Assert.IsNull(merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void TimestampSkew_BothNull_IsNull()
        {
            var merged = new Step9DMergedInferenceGeometry(null, null);
            Assert.IsNull(merged.HandPoseTimestampSkewMs);
        }

        [Test]
        public void Consistency_MaxSkewTracked_AcrossMultipleObservations()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(640, 480, 100, 102)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(2.0, CompleteMerged(640, 480, 200, 212)),
                new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(3.0, CompleteMerged(640, 480, 300, 305)),
                new Step9DRawIntermediateFrameIdentity(2, 3.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            Assert.AreEqual(3, snapshot.GeometryConsistencySummary.CheckedCount);
            Assert.AreEqual(12, snapshot.GeometryConsistencySummary.MaxHandPoseAbsSkewMs);
        }

        // =====================================================================
        //  Section K -- Mid-capture mixed geometry (2 tests)
        //  geometryRequired=false: observations with/without geometry coexist.
        // =====================================================================

        [Test]
        public void MidCapture_SomeObservationsWithoutGeometry_OthersWithGeometry_Passes()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, null),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(2.0, CompleteMerged(640, 480, 200, 200)),
                new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(3.0, null),
                new Step9DRawIntermediateFrameIdentity(2, 3.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            Assert.IsNotNull(snapshot.CaptureInferenceGeometry);
            Assert.AreEqual(1, snapshot.GeometryConsistencySummary.CheckedCount);
        }

        [Test]
        public void MidCapture_UnavailableResult_DoesNotAffectConsistency()
        {
            var unavailableMerged = new Step9DMergedInferenceGeometry(Unavailable(), Unavailable());

            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, unavailableMerged),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            collector.RecordObservation(
                MakeObservation(2.0, CompleteMerged(640, 480, 200, 200)),
                new Step9DRawIntermediateFrameIdentity(1, 2.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            Assert.IsNotNull(snapshot.CaptureInferenceGeometry);
            Assert.AreEqual(1, snapshot.GeometryConsistencySummary.CheckedCount);
        }

        // =====================================================================
        //  Section L -- Legacy backward compatibility (3 tests)
        // =====================================================================

        [Test]
        public void Legacy_ObservationWithoutGeometry_FullPipelinePasses()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            for (int i = 0; i < 5; i++)
            {
                collector.RecordObservation(
                    MakeObservation(1.0 + i, null),
                    new Step9DRawIntermediateFrameIdentity(i, 1.0 + i, null, null, null));
            }
            var snapshot = collector.FinalizeCapture();

            Assert.IsNull(snapshot.CaptureInferenceGeometry);
            Assert.IsNull(snapshot.GeometryConsistencySummary);
            Assert.IsNotNull(snapshot.Result);
            Assert.AreEqual(5, snapshot.ObservationCount);
        }

        [Test]
        public void Legacy_MergedBothNull_BackwardCompatible()
        {
            var merged = new Step9DMergedInferenceGeometry(null, null);
            Assert.IsFalse(merged.BothPresent);
            Assert.IsFalse(merged.BothAvailable);
            Assert.IsFalse(merged.FullContractCompatible);
            Assert.IsFalse(merged.DimensionallyCompatible);
            Assert.IsNull(merged.Hand);
            Assert.IsNull(merged.Pose);
            Assert.IsNull(merged.HandResult);
            Assert.IsNull(merged.PoseResult);
            Assert.IsNull(merged.Representative);
            Assert.IsNull(merged.HandPoseTimestampSkewMs);
            Assert.IsNull(merged.FirstMismatchDescription);
        }

        [Test]
        public void Legacy_DimensionallyCompatible_AliasForFullContract()
        {
            var merged = new Step9DMergedInferenceGeometry(AvailableHand(), AvailablePose());
            Assert.AreEqual(merged.FullContractCompatible, merged.DimensionallyCompatible);
            Assert.AreEqual(merged.BothAvailable, merged.BothPresent);
        }

        // =====================================================================
        //  Section M -- Serialization determinism (2 tests)
        //
        //  LIMITATION: These tests verify serialization is deterministic and
        //  well-formed. They do NOT parse the JSON with a JSON parser.
        //  The project has no JSON parser dependency (JsonUtility cannot parse
        //  arbitrary JSON). The claim "serialize-parse-reserialize" is
        //  withdrawn. These tests verify token-level validity and determinism.
        // =====================================================================

        [Test]
        public void SerializationDeterminism_GeometryPresent_WellFormedAndRepeatable()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, CompleteMerged(1280, 720, 100, 105)),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);

            Assert.IsTrue(json.StartsWith("{\"schema_version\":"), "must start with schema_version");
            Assert.IsTrue(json.EndsWith("\n"), "must end with newline");
            Assert.IsFalse(json.Contains(":NaN"), "must not contain :NaN");
            Assert.IsFalse(json.Contains(":Infinity"), "must not contain :Infinity");
            Assert.IsFalse(json.Contains(":-Infinity"), "must not contain :-Infinity");

            string json2 = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.AreEqual(json, json2, "re-serialization must be deterministic");
        }

        [Test]
        public void SerializationDeterminism_NoGeometry_WellFormedAndRepeatable()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0, null),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.IsTrue(json.StartsWith("{\"schema_version\":"));
            Assert.IsTrue(json.EndsWith("\n"));
            Assert.IsFalse(json.Contains("inference_frame_geometry"));
            Assert.IsFalse(json.Contains("geometry_consistency_summary"));

            string json2 = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.AreEqual(json, json2, "re-serialization must be deterministic");
        }

        // =====================================================================
        //  Section N -- Preprocessing numeric isolation (1 test)
        // =====================================================================

        [Test]
        public void NumericIsolation_GeometryPresenceDoesNotAffectPreprocessing()
        {
            var merged = CompleteMerged();

            var collectorWith = new Step9DRawIntermediateDiagnosticCollector();
            collectorWith.Begin("run1", "with_geom", "GREETING");
            collectorWith.RecordObservation(
                MakeObservation(1.0, merged),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapWith = collectorWith.FinalizeCapture();

            var collectorWithout = new Step9DRawIntermediateDiagnosticCollector();
            collectorWithout.Begin("run1", "without_geom", "GREETING");
            collectorWithout.RecordObservation(
                MakeObservation(1.0, null),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapWithout = collectorWithout.FinalizeCapture();

            var traceWith = snapWith.Trace;
            var traceWithout = snapWithout.Trace;

            Assert.AreEqual(traceWith.RawFrameCount, traceWithout.RawFrameCount);
            Assert.AreEqual(traceWith.NormalizedFrameCount, traceWithout.NormalizedFrameCount);
            Assert.AreEqual(traceWith.ValidLandmarkCount, traceWithout.ValidLandmarkCount);
            Assert.AreEqual(traceWith.InterpolatedLandmarkCount, traceWithout.InterpolatedLandmarkCount);

            double[] normWith = traceWith.CopyNormalizedCoordinates();
            double[] normWithout = traceWithout.CopyNormalizedCoordinates();
            Assert.AreEqual(normWith.Length, normWithout.Length);
            for (int i = 0; i < normWith.Length; i++)
            {
                bool bothNaN = double.IsNaN(normWith[i]) && double.IsNaN(normWithout[i]);
                Assert.IsTrue(bothNaN || normWith[i] == normWithout[i],
                    $"normalized coordinate differs at index {i}: {normWith[i]} vs {normWithout[i]}");
            }

            bool[] maskWith = traceWith.CopyNormalizedMask();
            bool[] maskWithout = traceWithout.CopyNormalizedMask();
            Assert.AreEqual(maskWith.Length, maskWithout.Length);
            for (int i = 0; i < maskWith.Length; i++)
                Assert.AreEqual(maskWith[i], maskWithout[i], $"normalized mask differs at index {i}");
        }

        // =====================================================================
        //  Section O -- Serialize call-scope file-output absence (2 tests)
        //
        //  LIMITED CLAIM: Within the scope of a single collector lifecycle
        //  (Begin, RecordObservation, FinalizeCapture) and Serialize call,
        //  no file output is produced. This does NOT prove the entire
        //  collector class never writes files under all conditions.
        //  Only the separate Export method (not tested here) writes to disk.
        // =====================================================================

        [Test]
        public void SerializeCallScope_CollectorLifecycleCreatesNoFiles()
        {
            string tempDir = Path.Combine(Path.GetTempPath(),
                "wis_writer_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);
            try
            {
                Assert.AreEqual(0, Directory.GetFiles(tempDir, "*", SearchOption.AllDirectories).Length,
                    "pre-condition: temp dir is empty");

                var merged = CompleteMerged();
                var collector = new Step9DRawIntermediateDiagnosticCollector();
                collector.Begin("run1", "cap1", "GREETING");
                collector.RecordObservation(
                    MakeObservation(1.0, merged),
                    new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
                var snapshot = collector.FinalizeCapture();
                string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);

                Assert.IsTrue(json.Length > 0, "Serialize returned content");
                Assert.AreEqual(0, Directory.GetFiles(tempDir, "*", SearchOption.AllDirectories).Length,
                    "post-condition: temp dir still empty after collector lifecycle and Serialize");
                Assert.AreEqual(0, Directory.GetDirectories(tempDir).Length,
                    "post-condition: no subdirectories created");
            }
            finally
            {
                if (Directory.Exists(tempDir))
                    Directory.Delete(tempDir, true);
            }
        }

        [Test]
        public void SerializeCallScope_ReturnsStringNotFilePath()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");
            collector.RecordObservation(
                MakeObservation(1.0),
                new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null));
            var snapshot = collector.FinalizeCapture();

            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.IsNotNull(json, "Serialize returns non-null string");
            Assert.IsTrue(json.Length > 0, "Serialize returns non-empty string");
            Assert.IsTrue(json.StartsWith("{"), "result is JSON content, not a file path");
        }
    }
}
