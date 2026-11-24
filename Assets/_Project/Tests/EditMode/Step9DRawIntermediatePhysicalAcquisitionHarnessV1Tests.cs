using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class Step9DRawIntermediatePhysicalAcquisitionHarnessV1Tests
    {
        readonly List<GameObject> createdObjects = new List<GameObject>();
        readonly HashSet<string> createdRunDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        [TearDown]
        public void TearDown()
        {
            foreach (GameObject value in createdObjects)
                if (value != null) UnityEngine.Object.DestroyImmediate(value);
            createdObjects.Clear();

            foreach (string directory in createdRunDirectories)
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            createdRunDirectories.Clear();
        }

        [Test]
        public void Arm_BlocksWhenBridgeIsUnassigned()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewUnassignedHarness();
            TrackRunDirectory(harness);

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Blocked, harness.HarnessState);
            Assert.IsFalse(harness.Arm());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_ASSIGNED, harness.LastFailureCode);
            Assert.AreEqual(harness.RunId, harness.FailureRunId);
            Assert.IsEmpty(harness.FailureCaptureId);
            Assert.AreEqual(0, harness.CaptureAttemptCount);
            Assert.IsFalse(Directory.Exists(harness.RunOutputDirectory));
        }

        [Test]
        public void InactiveAndUnarmedStates_PerformNoCaptureOrDiskIo()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out _);
            TrackRunDirectory(harness);
            Step9DRawIntermediateDiagnosticCollector collector = Collector(harness);

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Ready, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediateCollectorState.Idle, collector.State);
            Assert.AreEqual(0, harness.BufferedObservationCount);
            Assert.IsFalse(harness.BridgeSubscribed);
            Assert.IsFalse(Directory.Exists(harness.RunOutputDirectory));

            harness.enabled = false;
            InvokeOnDisable(harness);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Disabled, harness.HarnessState);
            Assert.AreEqual(0, harness.BufferedObservationCount);
            Assert.IsFalse(harness.BridgeSubscribed);
            Assert.IsFalse(Directory.Exists(harness.RunOutputDirectory));
        }

        [Test]
        public void TakeOrder_IsFixedGreetingLoveThankYou()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewUnassignedHarness();
            CollectionAssert.AreEqual(new[] { "GREETING", "LOVE", "THANK_YOU" }, harness.TakeOrder);
            Assert.AreEqual(1, harness.CurrentTakeNumber);
            Assert.AreEqual("GREETING", harness.CurrentExpectedWord);
        }

        [Test]
        public void SuccessfulTakes_UseUniqueRunAndCaptureIdsAndAdvanceInFixedOrder()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out _);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsNotEmpty(harness.RunId);

            string[] expectedWords = { "GREETING", "LOVE", "THANK_YOU" };
            var captureIds = new HashSet<string>(StringComparer.Ordinal);
            for (int take = 0; take < expectedWords.Length; take++)
            {
                Assert.AreEqual(take + 1, harness.CurrentTakeNumber);
                Assert.AreEqual(expectedWords[take], harness.CurrentExpectedWord);
                Assert.IsTrue(harness.StartCurrentTake());
                string captureId = harness.CurrentCaptureId;
                Assert.IsNotEmpty(captureId);
                Assert.IsTrue(captureIds.Add(captureId));
                RecordTwoFrames(harness, take * 10.0);

                Assert.IsTrue(harness.FinalizeAndExport());
                Assert.AreEqual(captureId, harness.LastCaptureId);
                Assert.IsTrue(harness.ChecksumVerified);
                Assert.Greater(harness.LastFileSize, 0L);
                Assert.AreEqual(64, harness.LastSha256.Length);
                Assert.IsTrue(File.Exists(harness.LastExportPath));
                Assert.IsTrue(Step9DRawIntermediateEvidenceJson.Verify(
                    harness.LastExportPath,
                    harness.LastExportPath + ".sha256"));
            }

            Assert.AreEqual(3, captureIds.Count);
            Assert.AreEqual(3, harness.CaptureAttemptCount);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed, harness.HarnessState);
            Assert.IsEmpty(harness.CurrentExpectedWord);
            Assert.IsEmpty(harness.CurrentCaptureId);
            Assert.IsFalse(harness.BridgeSubscribed);
        }

        [Test]
        public void FinalizeBeforeCaptureStart_IsBlocked()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out _);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            Assert.IsFalse(harness.FinalizeAndExport());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE, harness.LastFailureCode);
            Assert.AreEqual(1, harness.CurrentTakeNumber);
            Assert.AreEqual(0, harness.CaptureAttemptCount);
            Assert.IsFalse(Directory.Exists(harness.RunOutputDirectory));
        }

        [Test]
        public void EmptyCaptureFinalize_FailsWithoutRetryOrAdvance()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out _);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            string captureId = harness.CurrentCaptureId;

            Assert.IsFalse(harness.FinalizeAndExport());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.EMPTY_CAPTURE, harness.LastFailureCode);
            Assert.AreEqual(captureId, harness.CurrentCaptureId);
            Assert.AreEqual(captureId, harness.FailureCaptureId);
            Assert.AreEqual(1, harness.CurrentTakeNumber);
            Assert.AreEqual("GREETING", harness.CurrentExpectedWord);
            Assert.AreEqual(1, harness.CaptureAttemptCount);
            Assert.IsFalse(harness.StartCurrentTake());
            Assert.AreEqual(1, harness.CaptureAttemptCount);
            Assert.IsFalse(Directory.Exists(harness.RunOutputDirectory));
        }

        [Test]
        public void ExportFailure_DoesNotAdvanceAndDoesNotOverwriteTemporaryTarget()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out _);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            RecordTwoFrames(harness, 0.0);
            string temporaryPath = harness.PendingEvidencePath + ".tmp";
            Directory.CreateDirectory(Path.GetDirectoryName(temporaryPath));
            byte[] sentinel = { 11, 22, 33, 44 };
            File.WriteAllBytes(temporaryPath, sentinel);

            Assert.IsFalse(harness.FinalizeAndExport());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.OUTPUT_ALREADY_EXISTS, harness.LastFailureCode);
            Assert.AreEqual(1, harness.CurrentTakeNumber);
            Assert.AreEqual("GREETING", harness.CurrentExpectedWord);
            Assert.IsEmpty(harness.LastExportPath);
            CollectionAssert.AreEqual(sentinel, File.ReadAllBytes(temporaryPath));
        }

        [Test]
        public void FailureState_BlocksAdditionalCaptureAndPreservesFirstBridgeFailure()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out Step9CHandIdentityObservationBridge bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            string captureId = harness.CurrentCaptureId;

            RaiseBridgeFailure(bridge, new InvalidOperationException("synthetic bridge failure"));
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_FAILURE, harness.LastFailureCode);
            Assert.AreEqual("InvalidOperationException: synthetic bridge failure", harness.LastFailureMessage);
            Assert.AreEqual(harness.RunId, harness.FailureRunId);
            Assert.AreEqual(captureId, harness.FailureCaptureId);
            Assert.IsFalse(harness.BridgeSubscribed);

            Assert.IsFalse(harness.StartCurrentTake());
            Assert.IsFalse(harness.FinalizeAndExport());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_FAILURE, harness.LastFailureCode);
            Assert.AreEqual(1, harness.CaptureAttemptCount);
        }

        [Test]
        public void Disable_DisposesCollectorAndUnsubscribesBridgeEvents()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out Step9CHandIdentityObservationBridge bridge);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.BridgeSubscribed);
            Assert.AreEqual(1, EventSubscriberCount(bridge, "ObservationArrived"));
            Assert.AreEqual(1, EventSubscriberCount(bridge, "BridgeFailed"));

            harness.enabled = false;
            InvokeOnDisable(harness);

            Assert.IsFalse(harness.BridgeSubscribed);
            Assert.AreEqual(0, EventSubscriberCount(bridge, "ObservationArrived"));
            Assert.AreEqual(0, EventSubscriberCount(bridge, "BridgeFailed"));
            Assert.AreEqual(Step9DRawIntermediateCollectorState.Idle, Collector(harness).State);
        }

        [Test]
        public void ExistingEvidenceTarget_IsNeverOverwritten()
        {
            Step9DRawIntermediatePhysicalAcquisitionHarness harness = NewBoundHarness(out _);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            RecordTwoFrames(harness, 0.0);
            string evidencePath = harness.PendingEvidencePath;
            Directory.CreateDirectory(Path.GetDirectoryName(evidencePath));
            byte[] sentinel = { 90, 80, 70, 60, 50 };
            File.WriteAllBytes(evidencePath, sentinel);

            Assert.IsFalse(harness.FinalizeAndExport());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.OUTPUT_ALREADY_EXISTS, harness.LastFailureCode);
            CollectionAssert.AreEqual(sentinel, File.ReadAllBytes(evidencePath));
            Assert.IsFalse(File.Exists(evidencePath + ".sha256"));
            Assert.AreEqual(1, harness.CurrentTakeNumber);
            Assert.AreEqual("GREETING", harness.CurrentExpectedWord);
        }

        Step9DRawIntermediatePhysicalAcquisitionHarness NewUnassignedHarness()
        {
            var gameObject = new GameObject("Step9D-Harness-Test");
            createdObjects.Add(gameObject);
            return gameObject.AddComponent<Step9DRawIntermediatePhysicalAcquisitionHarness>();
        }

        Step9DRawIntermediatePhysicalAcquisitionHarness NewBoundHarness(
            out Step9CHandIdentityObservationBridge bridge)
        {
            var gameObject = new GameObject("Step9D-Harness-With-Bridge-Test");
            createdObjects.Add(gameObject);
            bridge = gameObject.AddComponent<Step9CHandIdentityObservationBridge>();
            SetBridgeBound(bridge, true);
            Step9DRawIntermediatePhysicalAcquisitionHarness harness =
                gameObject.AddComponent<Step9DRawIntermediatePhysicalAcquisitionHarness>();
            Assert.IsTrue(harness.AssignBridgeForDiagnostic(bridge));
            Assert.IsTrue(harness.BridgeAssigned);
            Assert.IsTrue(harness.BridgeBound);
            return harness;
        }

        void TrackRunDirectory(Step9DRawIntermediatePhysicalAcquisitionHarness harness)
        {
            createdRunDirectories.Add(harness.RunOutputDirectory);
        }

        static Step9DRawIntermediateDiagnosticCollector Collector(
            Step9DRawIntermediatePhysicalAcquisitionHarness harness)
        {
            FieldInfo field = typeof(Step9DRawIntermediatePhysicalAcquisitionHarness).GetField(
                "_collector",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field);
            var collector = field.GetValue(harness) as Step9DRawIntermediateDiagnosticCollector;
            Assert.IsNotNull(collector);
            return collector;
        }

        static void RecordTwoFrames(
            Step9DRawIntermediatePhysicalAcquisitionHarness harness,
            double firstTimestamp)
        {
            Step9DRawIntermediateDiagnosticCollector collector = Collector(harness);
            RawMediaPipeObservation first = Observation(firstTimestamp);
            RawMediaPipeObservation second = Observation(firstTimestamp + 0.03);
            collector.RecordObservation(
                first,
                Step9DRawIntermediateFrameIdentity.WithoutProviderIdentity(0, first.TimestampSeconds));
            collector.RecordObservation(
                second,
                Step9DRawIntermediateFrameIdentity.WithoutProviderIdentity(1, second.TimestampSeconds));
        }

        static RawMediaPipeObservation Observation(double timestamp)
        {
            var posePoints = new Dictionary<int, double[]>
            {
                { 11, Point(-1.0, 0.0, 0.0) },
                { 12, Point(1.0, 0.0, 0.0) },
                { 23, Point(-0.5, 1.0, 0.0) },
                { 24, Point(0.5, 1.0, 0.0) },
                { 13, Point(-1.2, 0.4, 0.0) },
                { 14, Point(1.2, 0.4, 0.0) },
                { 15, Point(-1.3, 0.8, 0.0) },
                { 16, Point(1.3, 0.8, 0.0) },
            };
            return new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.ApprovedStep7B,
                Set(33, posePoints),
                Set(21, new Dictionary<int, double[]>()),
                Set(21, new Dictionary<int, double[]>()),
                CompleteMerged());
        }

        static Step9DMergedInferenceGeometry CompleteMerged()
        {
            var hand = new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.ObservedAtHandInference,
                null, null, null,
                640, 480, 0, false,
                640, 480, 640, 480,
                "SRGBA",
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                null,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, 100, 1,
                "bottom-up");
            var pose = new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.ObservedAtPoseInference,
                null, null, null,
                640, 480, 0, false,
                640, 480, 640, 480,
                "SRGBA",
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                null,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                0L, 100, 1,
                "bottom-up");
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(hand),
                Step9DGeometryAcquisitionResult.ForAvailable(pose));
        }

        static RawLandmarkObservationSet Set(int count, Dictionary<int, double[]> points)
        {
            var coordinates = new double[count * 3];
            for (int index = 0; index < coordinates.Length; index++) coordinates[index] = double.NaN;
            var valid = new bool[count];
            var reappearance = new bool[count];
            var strong = new bool[count];
            foreach (KeyValuePair<int, double[]> pair in points)
            {
                valid[pair.Key] = true;
                Array.Copy(pair.Value, 0, coordinates, pair.Key * 3, 3);
            }
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, strong);
        }

        static double[] Point(double x, double y, double z) => new[] { x, y, z };

        static void InvokeOnDisable(
            Step9DRawIntermediatePhysicalAcquisitionHarness harness)
        {
            MethodInfo method = typeof(Step9DRawIntermediatePhysicalAcquisitionHarness).GetMethod(
                "OnDisable",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(method);
            method.Invoke(harness, null);
        }

        static void SetBridgeBound(Step9CHandIdentityObservationBridge bridge, bool value)
        {
            PropertyInfo property = typeof(Step9CHandIdentityObservationBridge).GetProperty(
                "IsBound",
                BindingFlags.Instance | BindingFlags.Public);
            Assert.IsNotNull(property);
            MethodInfo setter = property.GetSetMethod(true);
            Assert.IsNotNull(setter);
            setter.Invoke(bridge, new object[] { value });
        }

        static void RaiseBridgeFailure(Step9CHandIdentityObservationBridge bridge, Exception exception)
        {
            FieldInfo field = typeof(Step9CHandIdentityObservationBridge).GetField(
                "BridgeFailed",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field);
            var handler = field.GetValue(bridge) as Action<Exception>;
            Assert.IsNotNull(handler);
            handler.Invoke(exception);
        }

        static int EventSubscriberCount(Step9CHandIdentityObservationBridge bridge, string eventName)
        {
            FieldInfo field = typeof(Step9CHandIdentityObservationBridge).GetField(
                eventName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field);
            var handler = field.GetValue(bridge) as Delegate;
            return handler == null ? 0 : handler.GetInvocationList().Length;
        }
    }
}
