using System;
using System.Collections;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Core;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;
using Object = UnityEngine.Object;

namespace WordsInSilence.Tests.PlayMode
{
    public sealed class PhysicalWebcamCoordinateCompatibilityHarnessV1Tests
    {
        [UnityTest]
        public IEnumerator Prefab_LoadsInIsolationWithoutClaimingPhysicalDeviceReadiness()
        {
            GameObject prefab = Resources.Load<GameObject>("Diagnostics/PhysicalWebcamCoordinateCompatibility");
            Assert.IsNotNull(prefab);
            GameObject instance = Object.Instantiate(prefab);
            try
            {
                yield return null;
                yield return null;
                var harness = instance.GetComponent<PhysicalWebcamCoordinateCompatibilityHarness>();
                Assert.IsNotNull(harness);
                Assert.AreEqual(Step9CPhysicalCompatibilityConstants.RunId, harness.RunId);
                Assert.IsTrue(harness.RuntimeInitialized, harness.Status);
                Assert.IsNotNull(harness.ResumeReport);
                Assert.AreEqual(20, harness.ResumeReport.JsonFileCount);
                Assert.AreEqual(20, harness.ResumeReport.ChecksumFileCount);
                Assert.AreEqual(3, harness.ResumeReport.RecoveredValidCaptures.Count);
                Assert.AreEqual("GREETING", harness.ResumeReport.RecoveredValidCaptures[0].ExpectedWord);
                Assert.AreEqual("LOVE", harness.ResumeReport.RecoveredValidCaptures[1].ExpectedWord);
                Assert.AreEqual("THANK_YOU", harness.ResumeReport.RecoveredValidCaptures[2].ExpectedWord);
                Assert.AreEqual(string.Empty, harness.CurrentExpectedWord);
                Assert.AreEqual(0, harness.CurrentTakeNumber);
                StringAssert.Contains("Existing evidence contains all three verified fixed captures.", harness.Status);
                Assert.IsFalse(harness.PhysicalDeviceReadyClaimed);
                Assert.IsNull(harness.CurrentEvidence);
                Assert.IsNull(harness.LastReceipt);
                Assert.IsNotNull(instance.GetComponent<Step9CHandIdentityObservationBridge>());
            }
            finally
            {
                Object.Destroy(instance);
            }
        }

        [Test]
        public void Actual800By600ExpandedLayout_IsInBoundsAndSeparatesStatusFromPreview()
        {
            Step9CPanelLayout.CalculateExpanded(800, 600, out Rect header, out Rect status, out Rect preview);
            Rect screen = new Rect(0f, 0f, 800f, 600f);
            Assert.IsTrue(screen.Contains(header.min) && screen.Contains(header.max));
            Assert.IsTrue(screen.Contains(status.min) && screen.Contains(status.max));
            Assert.IsTrue(screen.Contains(preview.min) && screen.Contains(preview.max));
            Assert.LessOrEqual(status.xMax, preview.xMin);
            Assert.IsFalse(status.Overlaps(preview));
        }

        [UnityTest]
        public IEnumerator Step9CBridge_ReportsExactResolvedAndAmbiguousDuplicateDiagnostics()
        {
            var owner = new GameObject("Step9CBridgeRegression");
            var bridge = owner.AddComponent<Step9CHandIdentityObservationBridge>();
            try
            {
                yield return null;
                bridge.ResetObservationState();
                Step9CHandIdentityDiagnostic lastDiagnostic = null;
                RawMediaPipeObservation lastObservation = null;
                Exception lastFailure = null;
                bridge.HandIdentityDiagnosticObserved += value => lastDiagnostic = value;
                bridge.ObservationArrived += value => lastObservation = value;
                bridge.BridgeFailed += value => lastFailure = value;
                MethodInfo receive = typeof(Step9CHandIdentityObservationBridge).GetMethod("OnUnifiedFrame", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.IsNotNull(receive);

                receive.Invoke(bridge, new object[] { DuplicateFrame(1f, true, 0.2f, 0.8f, 0.2f, 0.8f) });
                Assert.AreEqual(1, bridge.ProviderFrameCount);
                Assert.AreEqual(1, bridge.ObservationCount);
                Assert.AreEqual(1, bridge.ResolvedDuplicateFrameCount);
                Assert.AreEqual(Step9CHandIdentityOutcome.ResolvedByPoseWrist, lastDiagnostic.Outcome);
                Assert.IsNotNull(lastObservation);
                Assert.IsNull(lastFailure);

                receive.Invoke(bridge, new object[] { DuplicateFrame(2f, false, 0.49f, 0.51f, 0.4f, 0.6f) });
                Assert.AreEqual(2, bridge.ProviderFrameCount);
                Assert.AreEqual(1, bridge.ObservationCount);
                Assert.AreEqual(1, bridge.AmbiguousDuplicateFrameCount);
                Assert.AreEqual(Step9CHandIdentityOutcome.Ambiguous, lastDiagnostic.Outcome);
                Assert.IsInstanceOf<Step9CAmbiguousHandIdentityException>(lastFailure);

                Step9CProviderBoundary beforeReset = bridge.SnapshotBoundary();
                Assert.AreEqual(2, beforeReset.ProviderFrameCount);
                Assert.AreEqual(2, beforeReset.LifetimeProviderFrameCount);
                Assert.AreEqual(1, beforeReset.LastLifetimeProviderFrameIndex);
                bridge.ResetObservationState();
                Step9CProviderBoundary afterReset = bridge.SnapshotBoundary();
                Assert.AreEqual(0, afterReset.ProviderFrameCount);
                Assert.AreEqual(2, afterReset.LifetimeProviderFrameCount, "Start resets capture-local counters only");
                Assert.AreEqual(1, afterReset.LastLifetimeProviderFrameIndex);
                Assert.That(afterReset.LastLifetimeProviderTimestampSeconds, Is.EqualTo(2.0).Within(1e-6));
            }
            finally
            {
                Object.Destroy(owner);
            }
        }

        static HandPoseFrame DuplicateFrame(float timestamp, bool left, float hand0X, float hand1X, float poseLeftX, float poseRightX)
        {
            var posePoints = new Vector3[33];
            var visibility = new float[33];
            posePoints[15] = new Vector3(poseLeftX, 0.5f, 0f);
            posePoints[16] = new Vector3(poseRightX, 0.5f, 0f);
            visibility[15] = visibility[16] = 1f;
            return new HandPoseFrame
            {
                Timestamp = timestamp,
                HandFrame = new LandmarkFrame
                {
                    Timestamp = timestamp,
                    Hands = new[] { Hand(left, hand0X, 0.9f), Hand(left, hand1X, 0.8f) },
                },
                PoseFrame = new UpperBodyPoseFrame { Timestamp = timestamp, AllPoints = posePoints, Visibility = visibility },
                HasPose = true,
            };
        }

        static HandLandmarks Hand(bool left, float x, float score)
        {
            var points = new LandmarkPoint[21];
            points[0] = new LandmarkPoint { X = x, Y = 0.5f, Z = 0f };
            return new HandLandmarks { IsLeftHand = left, Confidence = score, Points = points };
        }
    }
}
