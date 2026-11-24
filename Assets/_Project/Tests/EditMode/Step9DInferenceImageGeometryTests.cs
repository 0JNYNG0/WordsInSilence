using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    [TestFixture]
    public sealed class Step9DInferenceImageGeometryTests
    {
        [Test]
        public void Constructor_ValidValues_StoresAllProperties()
        {
            var g = new Step9DInferenceImageGeometry(640, 480, 0, false, 640, 480);

            Assert.AreEqual(640, g.WebCamTextureWidth);
            Assert.AreEqual(480, g.WebCamTextureHeight);
            Assert.AreEqual(0, g.VideoRotationAngle);
            Assert.IsFalse(g.VideoVerticallyMirrored);
            Assert.AreEqual(640, g.InferenceTextureWidth);
            Assert.AreEqual(480, g.InferenceTextureHeight);
        }

        [Test]
        public void Constructor_RotatedMirrored_StoresCorrectly()
        {
            var g = new Step9DInferenceImageGeometry(1920, 1080, 90, true, 1920, 1080);

            Assert.AreEqual(1920, g.WebCamTextureWidth);
            Assert.AreEqual(1080, g.WebCamTextureHeight);
            Assert.AreEqual(90, g.VideoRotationAngle);
            Assert.IsTrue(g.VideoVerticallyMirrored);
        }

        [Test]
        public void Constructor_ZeroWidth_Throws()
        {
            Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                new Step9DInferenceImageGeometry(0, 480, 0, false, 640, 480));
        }

        [Test]
        public void Constructor_NegativeHeight_Throws()
        {
            Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                new Step9DInferenceImageGeometry(640, -1, 0, false, 640, 480));
        }

        [Test]
        public void Constructor_ZeroInferenceWidth_Throws()
        {
            Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                new Step9DInferenceImageGeometry(640, 480, 0, false, 0, 480));
        }

        [Test]
        public void Clone_ReturnsEqualButDistinctInstance()
        {
            var original = new Step9DInferenceImageGeometry(640, 480, 90, true, 640, 480);
            var cloned = original.Clone();

            Assert.AreNotSame(original, cloned);
            Assert.AreEqual(original.WebCamTextureWidth, cloned.WebCamTextureWidth);
            Assert.AreEqual(original.WebCamTextureHeight, cloned.WebCamTextureHeight);
            Assert.AreEqual(original.VideoRotationAngle, cloned.VideoRotationAngle);
            Assert.AreEqual(original.VideoVerticallyMirrored, cloned.VideoVerticallyMirrored);
            Assert.AreEqual(original.InferenceTextureWidth, cloned.InferenceTextureWidth);
            Assert.AreEqual(original.InferenceTextureHeight, cloned.InferenceTextureHeight);
        }

        [Test]
        public void Snapshot_WithGeometry_PreservesGeometry()
        {
            var geometry = new Step9DInferenceImageGeometry(640, 480, 0, false, 640, 480);

            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", geometry);

            var obs = MakeObservation(1.0);
            var identity = new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null);
            collector.RecordObservation(obs, identity);

            var snapshot = collector.FinalizeCapture();

            Assert.IsNotNull(snapshot.InferenceImageGeometry);
            Assert.AreEqual(640, snapshot.InferenceImageGeometry.WebCamTextureWidth);
            Assert.AreEqual(480, snapshot.InferenceImageGeometry.WebCamTextureHeight);
            Assert.AreEqual(0, snapshot.InferenceImageGeometry.VideoRotationAngle);
            Assert.IsFalse(snapshot.InferenceImageGeometry.VideoVerticallyMirrored);
        }

        [Test]
        public void Snapshot_WithoutGeometry_GeometryIsNull()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");

            var obs = MakeObservation(1.0);
            var identity = new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null);
            collector.RecordObservation(obs, identity);

            var snapshot = collector.FinalizeCapture();

            Assert.IsNull(snapshot.InferenceImageGeometry);
        }

        [Test]
        public void Serialization_WithGeometry_ContainsGeometryFields()
        {
            var geometry = new Step9DInferenceImageGeometry(1280, 720, 0, false, 1280, 720);

            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", geometry);

            var obs = MakeObservation(1.0);
            var identity = new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null);
            collector.RecordObservation(obs, identity);

            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);

            StringAssert.Contains("\"inference_image_geometry\":{", json);
            StringAssert.Contains("\"webcam_texture_width\":1280", json);
            StringAssert.Contains("\"webcam_texture_height\":720", json);
            StringAssert.Contains("\"video_rotation_angle\":0", json);
            StringAssert.Contains("\"video_vertically_mirrored\":false", json);
            StringAssert.Contains("\"inference_texture_width\":1280", json);
            StringAssert.Contains("\"inference_texture_height\":720", json);
        }

        [Test]
        public void Serialization_WithoutGeometry_OmitsGeometryBlock()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING");

            var obs = MakeObservation(1.0);
            var identity = new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null);
            collector.RecordObservation(obs, identity);

            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);

            Assert.IsFalse(json.Contains("inference_image_geometry"), "geometry block should be absent when not supplied");
        }

        [Test]
        public void Serialization_MirroredRotated_SerializesCorrectly()
        {
            var geometry = new Step9DInferenceImageGeometry(1920, 1080, 270, true, 1920, 1080);

            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "LOVE", geometry);

            var obs = MakeObservation(1.0);
            var identity = new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null);
            collector.RecordObservation(obs, identity);

            var snapshot = collector.FinalizeCapture();
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);

            StringAssert.Contains("\"video_rotation_angle\":270", json);
            StringAssert.Contains("\"video_vertically_mirrored\":true", json);
        }

        [Test]
        public void Cancel_ClearsGeometry()
        {
            var geometry = new Step9DInferenceImageGeometry(640, 480, 0, false, 640, 480);

            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run1", "cap1", "GREETING", geometry);
            collector.Cancel();

            // Begin again without geometry
            collector.Begin("run1", "cap2", "LOVE");

            var obs = MakeObservation(1.0);
            var identity = new Step9DRawIntermediateFrameIdentity(0, 1.0, null, null, null);
            collector.RecordObservation(obs, identity);

            var snapshot = collector.FinalizeCapture();
            Assert.IsNull(snapshot.InferenceImageGeometry);
        }

        static RawMediaPipeObservation MakeObservation(double timestamp)
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
                MakeSet(21, new Dictionary<int, double[]>()));
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
    }
}
