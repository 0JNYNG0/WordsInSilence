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
    /// <summary>
    /// Step 10O v1.3c1 - Transient typed-ambiguity tolerance tests with:
    /// - Geometry-carrying observations (640x480, BothAvailable, FullContractCompatible)
    /// - Bridge-to-collector observation flow (not direct collector insertion)
    /// - Exported JSON provenance verification
    /// - Corrected counter semantics (tolerated drop count, not occurrence count)
    /// </summary>
    public sealed class Step9DTransientAmbiguityToleranceV1Tests
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

        // =====================================================================
        //  1. Bridge-to-collector observation flow
        // =====================================================================

        [Test]
        public void BridgeCollectorFlow_ObservationArrivesThroughBridge()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            Assert.AreEqual(0, harness.BufferedObservationCount);

            RaiseBridgeObservation(bridge, GeometryObservation(1.0), 1, 1, 0L);
            Assert.AreEqual(1, harness.BufferedObservationCount);

            RaiseBridgeObservation(bridge, GeometryObservation(1.03), 2, 2, 1L);
            Assert.AreEqual(2, harness.BufferedObservationCount);
        }

        // =====================================================================
        //  2-6. Core tolerance behavior
        // =====================================================================

        [Test]
        public void Capturing_FirstAmbiguity_IsNonfatal_StateRemainsCapturing()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, Ambiguity(100));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONE, harness.LastFailureCode);
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);
            Assert.IsTrue(harness.BridgeSubscribed);
        }

        [Test]
        public void Capturing_SecondAmbiguity_IsFatal_StateBecomesFailedWithBridgeFailure()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, Ambiguity(100));
            RaiseBridgeFailure(bridge, Ambiguity(101));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_FAILURE, harness.LastFailureCode);
            Assert.IsFalse(harness.BridgeSubscribed);
        }

        [Test]
        public void Capturing_SecondAmbiguity_ToleratedCountStaysAtOne()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, Ambiguity(100));
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);

            RaiseBridgeFailure(bridge, Ambiguity(101));
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);
        }

        [Test]
        public void Capturing_GenericException_ImmediatelyFatal_NoTolerance()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, new InvalidOperationException("generic bridge failure"));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_FAILURE, harness.LastFailureCode);
            Assert.AreEqual(0, harness.ToleratedDropCountThisTake);
            Assert.IsFalse(harness.BridgeSubscribed);
        }

        [Test]
        public void Capturing_NullException_ImmediatelyFatal()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, null);

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_FAILURE, harness.LastFailureCode);
            Assert.IsTrue(harness.LastFailureMessage.Contains("unknown failure"));
        }

        // =====================================================================
        //  7. Diagnostic exception fatality
        // =====================================================================

        [Test]
        public void Capturing_DiagnosticException_ImmediatelyFatal_MapsCorrectCode()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, new Step9DRawIntermediateDiagnosticException(
                Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                "geometry required but observation has no merged geometry"));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.INFERENCE_GEOMETRY_MISSING, harness.LastFailureCode);
            Assert.AreEqual(0, harness.ToleratedDropCountThisTake);
        }

        // =====================================================================
        //  8-9. Drop provenance
        // =====================================================================

        [Test]
        public void FirstAmbiguity_RecordsDropProvenance_AllFields()
        {
            var harness = NewCapturingHarness(out var bridge);
            string captureId = harness.CurrentCaptureId;
            RaiseBridgeFailure(bridge, Ambiguity(200, "Right", 0.90, "Right", 0.99, "best_pose_wrist_assignment_too_far"));

            Assert.AreEqual(1, harness.AmbiguityDrops.Count);
            var drop = harness.AmbiguityDrops[0];
            Assert.AreEqual(captureId, drop.CaptureId);
            Assert.AreEqual(1, drop.DropOrdinal);
            Assert.AreEqual(200, drop.ProviderFrameIndex);
            Assert.AreEqual(200.0 / 30.0, drop.ProviderTimestampSeconds, 1e-6);
            Assert.AreEqual("Right", drop.Category0);
            Assert.AreEqual(0.90, drop.Score0, 1e-6);
            Assert.AreEqual("Right", drop.Category1);
            Assert.AreEqual(0.99, drop.Score1, 1e-6);
            Assert.AreEqual("best_pose_wrist_assignment_too_far", drop.Reason);
        }

        [Test]
        public void SecondAmbiguity_DoesNotAddSecondDropRecord()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, Ambiguity(200));
            RaiseBridgeFailure(bridge, Ambiguity(201));

            Assert.AreEqual(1, harness.AmbiguityDrops.Count);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
        }

        // =====================================================================
        //  10-11. Bridge subscriptions and finalize after tolerance
        // =====================================================================

        [Test]
        public void AfterFirstAmbiguity_BridgeSubscriptionsIntact()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, Ambiguity(300));

            Assert.IsTrue(harness.BridgeSubscribed);
            Assert.AreEqual(1, EventSubscriberCount(bridge, "ObservationArrived"));
            Assert.AreEqual(1, EventSubscriberCount(bridge, "BridgeFailed"));
        }

        [Test]
        public void AfterFirstAmbiguity_FinalizeSucceedsNormally()
        {
            var harness = NewCapturingHarness(out var bridge);
            int fc = RecordTwoFramesViaBridge(bridge, 5.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(400), fc + 1, (long)fc);
            Assert.AreEqual(2, harness.BufferedObservationCount);

            // Post-drop: valid observations continue
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 5.1, fc);
            Assert.AreEqual(4, harness.BufferedObservationCount);

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing, harness.HarnessState);
            Assert.IsTrue(harness.FinalizeAndExport());
            Assert.IsTrue(harness.ChecksumVerified);
            Assert.Greater(harness.LastFileSize, 0L);
        }

        // =====================================================================
        //  11b. Post-drop continuation: full lifecycle verification
        // =====================================================================

        [Test]
        public void PostDropContinuation_ValidObservationsAfterAmbiguity_ContinueCaptureAndExportCorrectIdentity()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            string captureId = harness.CurrentCaptureId;

            // Step 1: valid observation at provider frame 99
            RaiseBridgeObservation(bridge, GeometryObservation(99.0 / 30.0), 100, 1, 99L);
            Assert.AreEqual(1, harness.BufferedObservationCount);

            // Step 2: ambiguity at provider frame 100
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(100), 101, 100L);

            // Step 3: verify post-drop state
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONE, harness.LastFailureCode);
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);
            Assert.IsTrue(harness.BridgeSubscribed);
            Assert.AreEqual(1, harness.BufferedObservationCount, "Dropped frame must not be buffered");

            // Step 4: valid observations at provider frames 101, 102 AFTER the ambiguity
            RaiseBridgeObservation(bridge, GeometryObservation(101.0 / 30.0), 102, 2, 101L);
            RaiseBridgeObservation(bridge, GeometryObservation(102.0 / 30.0), 103, 3, 102L);
            Assert.AreEqual(3, harness.BufferedObservationCount, "Post-drop observations must be buffered");

            // Step 5: finalize and export
            Assert.IsTrue(harness.FinalizeAndExport());
            Assert.IsTrue(harness.ChecksumVerified);
            Assert.IsTrue(Step9DRawIntermediateEvidenceJson.Verify(
                harness.LastExportPath, harness.LastExportPath + ".sha256"));

            // Step 6: read and verify exported JSON
            string json = File.ReadAllText(harness.LastExportPath);

            // Saved observations: provider frames 99, 101, 102
            Assert.IsTrue(json.Contains("\"provider_frame_index\":99"), "Frame 99 must be in identity");
            Assert.IsTrue(json.Contains("\"provider_frame_index\":101"), "Frame 101 must be in identity");
            Assert.IsTrue(json.Contains("\"provider_frame_index\":102"), "Frame 102 must be in identity");

            // Dropped frame 100 must NOT appear in frame_identity
            // (it should only appear in transient_hand_identity_drops)
            int dropsSectionStart = json.IndexOf("\"transient_hand_identity_drops\":[");
            int frameIdentityStart = json.IndexOf("\"frame_identity\":[");
            Assert.Greater(dropsSectionStart, 0);
            Assert.Greater(frameIdentityStart, 0);
            string frameIdentitySection = json.Substring(frameIdentityStart);
            // provider_frame_index:100 should NOT be in frame_identity section
            // but SHOULD be in drops section
            string dropsSection = json.Substring(dropsSectionStart, frameIdentityStart - dropsSectionStart);
            Assert.IsTrue(dropsSection.Contains("\"provider_frame_index\":100"),
                "Dropped frame 100 must be in transient_hand_identity_drops");
            Assert.IsFalse(frameIdentitySection.Contains("\"provider_frame_index\":100"),
                "Dropped frame 100 must NOT be in frame_identity");

            // Exactly one drop record
            Assert.IsTrue(json.Contains("\"drop_ordinal\":1"), "Drop ordinal must be 1");
            Assert.IsTrue(json.Contains("\"capture_id\":\"" + captureId + "\""),
                "Drop must reference the current capture ID");

            // Verify timestamps strictly increasing in frame_identity
            // Parse frame indices from frame_identity section
            int count99 = CountOccurrences(frameIdentitySection, "\"provider_frame_index\":99");
            int count101 = CountOccurrences(frameIdentitySection, "\"provider_frame_index\":101");
            int count102 = CountOccurrences(frameIdentitySection, "\"provider_frame_index\":102");
            Assert.AreEqual(1, count99, "Frame 99 should appear exactly once");
            Assert.AreEqual(1, count101, "Frame 101 should appear exactly once");
            Assert.AreEqual(1, count102, "Frame 102 should appear exactly once");

            // Verify 99 < 101 < 102 ordering (positional check)
            int pos99 = frameIdentitySection.IndexOf("\"provider_frame_index\":99");
            int pos101 = frameIdentitySection.IndexOf("\"provider_frame_index\":101");
            int pos102 = frameIdentitySection.IndexOf("\"provider_frame_index\":102");
            Assert.Less(pos99, pos101, "Frame 99 must precede frame 101");
            Assert.Less(pos101, pos102, "Frame 101 must precede frame 102");
        }

        // =====================================================================
        //  11c. Three-take with take-local ordinals and JSON verification
        // =====================================================================

        [Test]
        public void ThreeTakes_TakeLocalOrdinals_EachJsonContainsOnlyOwnDrop()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            string[] words = { "GREETING", "LOVE", "THANK_YOU" };
            var paths = new List<string>();
            var captureIds = new List<string>();
            int fc = 0;
            for (int i = 0; i < words.Length; i++)
            {
                Assert.IsTrue(harness.StartCurrentTake());
                captureIds.Add(harness.CurrentCaptureId);
                fc = RecordTwoFramesViaBridge(bridge, i * 10.0, fc);
                RaiseBridgeFailureWithProviderState(bridge, Ambiguity(1100 + i), fc + 1, (long)fc);
                fc++;
                fc = RecordTwoFramesViaBridge(bridge, i * 10.0 + 0.2, fc);
                Assert.IsTrue(harness.FinalizeAndExport());
                paths.Add(harness.LastExportPath);
            }
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed, harness.HarnessState);

            // Verify each JSON has take-local ordinal = 1 and only its own capture ID
            for (int i = 0; i < words.Length; i++)
            {
                string json = File.ReadAllText(paths[i]);
                Assert.IsTrue(json.Contains("\"drop_ordinal\":1"),
                    words[i] + " drop ordinal must be 1 (take-local)");
                Assert.IsTrue(json.Contains("\"capture_id\":\"" + captureIds[i] + "\""),
                    words[i] + " must reference its own capture ID");
                // Must NOT contain other takes' capture IDs in the drops section
                for (int j = 0; j < words.Length; j++)
                {
                    if (j == i) continue;
                    int dropsStart = json.IndexOf("\"transient_hand_identity_drops\":[");
                    int dropsEnd = json.IndexOf("]", dropsStart + 35);
                    string dropsSlice = json.Substring(dropsStart, dropsEnd - dropsStart);
                    Assert.IsFalse(dropsSlice.Contains(captureIds[j]),
                        words[i] + " must not contain " + words[j] + " capture ID in drops");
                }
                Assert.IsTrue(json.Contains("\"transient_hand_identity_drops\":[{"),
                    words[i] + " must have non-empty drops");
            }

            // All three capture IDs are distinct
            Assert.AreNotEqual(captureIds[0], captureIds[1]);
            Assert.AreNotEqual(captureIds[1], captureIds[2]);
            Assert.AreNotEqual(captureIds[0], captureIds[2]);

            // Ambiguity-free check: verify empty drops for a take without ambiguity
            // (This is already covered by ThreeTakes_NoAmbiguity_AllExportsHaveEmptyDrops)
        }

        // =====================================================================
        //  12-13. Per-take counter reset
        // =====================================================================

        [Test]
        public void NewTake_ResetsToleratedDropCounter()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            int fc = RecordTwoFramesViaBridge(bridge, 0.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(500), fc + 1, (long)fc);
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 0.2, fc);

            Assert.IsTrue(harness.FinalizeAndExport());

            Assert.IsTrue(harness.StartCurrentTake());
            Assert.AreEqual(0, harness.ToleratedDropCountThisTake);
        }

        [Test]
        public void SecondTake_AllowsOwnFirstAmbiguity()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            // Take 1: valid -> ambiguity -> valid -> finalize
            Assert.IsTrue(harness.StartCurrentTake());
            int fc = RecordTwoFramesViaBridge(bridge, 0.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(600), fc + 1, (long)fc);
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 0.2, fc);
            Assert.IsTrue(harness.FinalizeAndExport());

            // Take 2: gets its own tolerance
            Assert.IsTrue(harness.StartCurrentTake());
            Assert.AreEqual(0, harness.ToleratedDropCountThisTake);
            fc = RecordTwoFramesViaBridge(bridge, 10.0, fc);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(601), fc + 1, (long)fc);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing, harness.HarnessState);
            Assert.AreEqual(1, harness.ToleratedDropCountThisTake);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 10.2, fc);
        }

        // =====================================================================
        //  14. Drops accumulate across takes with sequential ordinals
        // =====================================================================

        [Test]
        public void DropsAccumulateAcrossTakes_TakeLocalOrdinal()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            // Take 1: valid -> ambiguity -> valid -> finalize
            Assert.IsTrue(harness.StartCurrentTake());
            string captureId1 = harness.CurrentCaptureId;
            int fc = RecordTwoFramesViaBridge(bridge, 0.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(700), fc + 1, (long)fc);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 0.2, fc);
            Assert.IsTrue(harness.FinalizeAndExport());

            // Take 2: valid -> ambiguity -> valid -> finalize
            Assert.IsTrue(harness.StartCurrentTake());
            string captureId2 = harness.CurrentCaptureId;
            fc = RecordTwoFramesViaBridge(bridge, 10.0, fc);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(701), fc + 1, (long)fc);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 10.2, fc);
            Assert.IsTrue(harness.FinalizeAndExport());

            Assert.AreEqual(2, harness.AmbiguityDrops.Count);
            Assert.AreEqual(1, harness.AmbiguityDrops[0].DropOrdinal, "Take-local ordinal for take 1");
            Assert.AreEqual(1, harness.AmbiguityDrops[1].DropOrdinal, "Take-local ordinal for take 2");
            Assert.AreEqual(captureId1, harness.AmbiguityDrops[0].CaptureId);
            Assert.AreEqual(captureId2, harness.AmbiguityDrops[1].CaptureId);
            Assert.AreNotEqual(captureId1, captureId2, "Capture IDs must differ between takes");
        }

        // =====================================================================
        //  15. Ambiguity in non-Capturing states
        // =====================================================================

        [Test]
        public void Ambiguity_WhenArmed_IsIgnored()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed, harness.HarnessState);

            RaiseBridgeFailure(bridge, Ambiguity(800));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONE, harness.LastFailureCode);
            Assert.AreEqual(0, harness.ToleratedDropCountThisTake);
        }

        // =====================================================================
        //  16. Full 3-take completion with tolerances
        // =====================================================================

        [Test]
        public void ThreeTakes_WithOneToleranceEach_CompletesSuccessfully()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            string[] words = { "GREETING", "LOVE", "THANK_YOU" };
            var captureIds = new List<string>();
            int fc = 0;
            for (int i = 0; i < words.Length; i++)
            {
                Assert.AreEqual(words[i], harness.CurrentExpectedWord);
                Assert.IsTrue(harness.StartCurrentTake());
                captureIds.Add(harness.CurrentCaptureId);
                // valid -> ambiguity -> valid -> finalize
                fc = RecordTwoFramesViaBridge(bridge, i * 10.0, fc);
                RaiseBridgeFailureWithProviderState(bridge, Ambiguity(900 + i), fc + 1, (long)fc);
                fc++;
                Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing, harness.HarnessState);
                fc = RecordTwoFramesViaBridge(bridge, i * 10.0 + 0.2, fc);
                Assert.IsTrue(harness.FinalizeAndExport());
                Assert.IsTrue(harness.ChecksumVerified);
            }

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed, harness.HarnessState);
            Assert.AreEqual(3, harness.AmbiguityDrops.Count);
            Assert.IsFalse(harness.BridgeSubscribed);

            // Take-local ordinals: all must be 1
            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(1, harness.AmbiguityDrops[i].DropOrdinal, "Take-local ordinal for " + words[i]);
                Assert.AreEqual(captureIds[i], harness.AmbiguityDrops[i].CaptureId);
            }
            // All capture IDs distinct
            Assert.AreNotEqual(captureIds[0], captureIds[1]);
            Assert.AreNotEqual(captureIds[1], captureIds[2]);
            Assert.AreNotEqual(captureIds[0], captureIds[2]);
        }

        // =====================================================================
        //  17-19. Exported JSON verification
        // =====================================================================

        [Test]
        public void ExportedJson_ContainsTransientHandIdentityDropsField()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            int fc = RecordTwoFramesViaBridge(bridge, 0.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(200), fc + 1, (long)fc);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 0.2, fc);

            Assert.IsTrue(harness.FinalizeAndExport());
            string json = File.ReadAllText(harness.LastExportPath);
            Assert.IsTrue(json.Contains("\"transient_hand_identity_drops\":["), "JSON must contain drops field");
            Assert.IsTrue(json.Contains("\"capture_id\""), "Drop must contain capture_id");
            Assert.IsTrue(json.Contains("\"provider_frame_index\":200"), "Drop must contain correct provider_frame_index");
            Assert.IsTrue(json.Contains("\"drop_ordinal\":1"), "Drop must have ordinal 1");
            Assert.IsTrue(json.Contains("\"reason\":"), "Drop must contain reason");
        }

        [Test]
        public void ExportedJson_DropsFieldIsEmptyArrayWhenNoAmbiguity()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            RecordTwoFramesViaBridge(bridge, 0.0, 0);

            Assert.IsTrue(harness.FinalizeAndExport());
            string json = File.ReadAllText(harness.LastExportPath);
            Assert.IsTrue(json.Contains("\"transient_hand_identity_drops\":[]"),
                "JSON must have empty drops array when no ambiguity occurred");
        }

        [Test]
        public void ExportedJson_ChecksumCoversProvenance()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            int fc = RecordTwoFramesViaBridge(bridge, 0.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(200), fc + 1, (long)fc);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 0.2, fc);

            Assert.IsTrue(harness.FinalizeAndExport());
            Assert.IsTrue(harness.ChecksumVerified);
            Assert.IsTrue(Step9DRawIntermediateEvidenceJson.Verify(
                harness.LastExportPath, harness.LastExportPath + ".sha256"));
        }

        // =====================================================================
        //  20-21. Geometry fatality
        // =====================================================================

        [Test]
        public void GeometryMissingException_IsFatal()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, new Step9DRawIntermediateDiagnosticException(
                Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                "geometry required but observation has no merged geometry"));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.INFERENCE_GEOMETRY_MISSING, harness.LastFailureCode);
        }

        [Test]
        public void GeometryContractChangedException_IsFatal()
        {
            var harness = NewCapturingHarness(out var bridge);
            RaiseBridgeFailure(bridge, new Step9DRawIntermediateDiagnosticException(
                Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED,
                "hand geometry contract changed mid-capture"));

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed, harness.HarnessState);
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED, harness.LastFailureCode);
        }

        // =====================================================================
        //  22. Per-take drops in export
        // =====================================================================

        [Test]
        public void PerTakeDrops_OnlyInThatTakeExport()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            // Take 1: valid -> ambiguity -> valid -> finalize
            Assert.IsTrue(harness.StartCurrentTake());
            int fc = RecordTwoFramesViaBridge(bridge, 0.0, 0);
            RaiseBridgeFailureWithProviderState(bridge, Ambiguity(1000), fc + 1, (long)fc);
            fc++;
            fc = RecordTwoFramesViaBridge(bridge, 0.2, fc);
            Assert.IsTrue(harness.FinalizeAndExport());
            string take1Path = harness.LastExportPath;

            // Take 2: no ambiguity
            Assert.IsTrue(harness.StartCurrentTake());
            fc = RecordTwoFramesViaBridge(bridge, 10.0, fc);
            Assert.IsTrue(harness.FinalizeAndExport());
            string take2Path = harness.LastExportPath;

            string json1 = File.ReadAllText(take1Path);
            string json2 = File.ReadAllText(take2Path);

            Assert.IsTrue(json1.Contains("\"drop_ordinal\":1"), "Take 1 must have drop");
            Assert.IsTrue(json2.Contains("\"transient_hand_identity_drops\":[]"),
                "Take 2 must have empty drops (ambiguity was in take 1)");
        }

        // =====================================================================
        //  23. Three takes, no ambiguity, all exports have empty drops
        // =====================================================================

        [Test]
        public void ThreeTakes_NoAmbiguity_AllExportsHaveEmptyDrops()
        {
            var harness = NewBoundHarness(out var bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());

            int fc = 0;
            var paths = new List<string>();
            for (int i = 0; i < 3; i++)
            {
                Assert.IsTrue(harness.StartCurrentTake());
                fc = RecordTwoFramesViaBridge(bridge, i * 10.0, fc);
                Assert.IsTrue(harness.FinalizeAndExport());
                paths.Add(harness.LastExportPath);
            }

            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed, harness.HarnessState);
            Assert.AreEqual(0, harness.AmbiguityDrops.Count);

            foreach (string path in paths)
            {
                string json = File.ReadAllText(path);
                Assert.IsTrue(json.Contains("\"transient_hand_identity_drops\":[]"),
                    "All takes must have empty drops array: " + Path.GetFileName(path));
            }
        }

        // =====================================================================
        //  Helpers
        // =====================================================================

        Step9DRawIntermediatePhysicalAcquisitionHarness NewBoundHarness(
            out Step9CHandIdentityObservationBridge bridge)
        {
            var gameObject = new GameObject("Step9D-Tolerance-With-Bridge-Test");
            createdObjects.Add(gameObject);
            bridge = gameObject.AddComponent<Step9CHandIdentityObservationBridge>();
            SetBridgeBound(bridge, true);
            var harness = gameObject.AddComponent<Step9DRawIntermediatePhysicalAcquisitionHarness>();
            Assert.IsTrue(harness.AssignBridgeForDiagnostic(bridge));
            return harness;
        }

        Step9DRawIntermediatePhysicalAcquisitionHarness NewCapturingHarness(
            out Step9CHandIdentityObservationBridge bridge)
        {
            var harness = NewBoundHarness(out bridge);
            TrackRunDirectory(harness);
            Assert.IsTrue(harness.Arm());
            Assert.IsTrue(harness.StartCurrentTake());
            Assert.AreEqual(Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing, harness.HarnessState);
            return harness;
        }

        void TrackRunDirectory(Step9DRawIntermediatePhysicalAcquisitionHarness harness)
        {
            createdRunDirectories.Add(harness.RunOutputDirectory);
        }

        /// <summary>
        /// Records two geometry-carrying observations through bridge -> collector flow.
        /// Returns the updated bridge frame counter for subsequent calls.
        /// </summary>
        static int RecordTwoFramesViaBridge(
            Step9CHandIdentityObservationBridge bridge,
            double firstTimestamp,
            int bridgeFrameCounter)
        {
            bridgeFrameCounter++;
            RaiseBridgeObservation(bridge, GeometryObservation(firstTimestamp),
                bridgeFrameCounter, bridgeFrameCounter, (long)(bridgeFrameCounter - 1));
            bridgeFrameCounter++;
            RaiseBridgeObservation(bridge, GeometryObservation(firstTimestamp + 0.03),
                bridgeFrameCounter, bridgeFrameCounter, (long)(bridgeFrameCounter - 1));
            return bridgeFrameCounter;
        }

        static void RaiseBridgeObservation(
            Step9CHandIdentityObservationBridge bridge,
            RawMediaPipeObservation observation,
            int providerFrameCount,
            int observationCount,
            long lastLifetimeProviderFrameIndex)
        {
            SetBridgeProperty(bridge, "ProviderFrameCount", providerFrameCount);
            SetBridgeProperty(bridge, "ObservationCount", observationCount);
            SetBridgeProperty(bridge, "LastLifetimeProviderFrameIndex", lastLifetimeProviderFrameIndex);

            FieldInfo field = typeof(Step9CHandIdentityObservationBridge).GetField(
                "ObservationArrived",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, "ObservationArrived backing field not found");
            var handler = field.GetValue(bridge) as Action<RawMediaPipeObservation>;
            Assert.IsNotNull(handler, "No subscribers on ObservationArrived");
            handler.Invoke(observation);
        }

        static RawMediaPipeObservation GeometryObservation(double timestamp)
        {
            var merged = CompleteMerged();
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
                Set(33, posePoints),
                Set(21, new Dictionary<int, double[]>()),
                Set(21, new Dictionary<int, double[]>()),
                merged);
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

        static Step9CAmbiguousHandIdentityException Ambiguity(int providerFrameIndex,
            string cat0 = "Left", double score0 = 0.91,
            string cat1 = "Left", double score1 = 0.99,
            string reason = "same_frame_pose_wrists_unavailable_or_low_visibility")
        {
            double timestamp = providerFrameIndex / 30.0;
            var diagnostic = new Step9CHandIdentityDiagnostic(
                providerFrameIndex: providerFrameIndex,
                providerTimestampSeconds: timestamp,
                handFrameTimestampSeconds: timestamp,
                sourceHandCount: 2,
                sourceCategory0: cat0,
                sourceScore0: score0,
                sourceCategory1: cat1,
                sourceScore1: score1,
                hand0WristX: 0.25,
                hand0WristY: 0.50,
                hand1WristX: 0.75,
                hand1WristY: 0.50,
                poseLeftWristX: double.NaN,
                poseLeftWristY: double.NaN,
                poseRightWristX: double.NaN,
                poseRightWristY: double.NaN,
                poseLeftWristVisibility: double.NaN,
                poseRightWristVisibility: double.NaN,
                directAssignmentCost: double.NaN,
                swappedAssignmentCost: double.NaN,
                bestAssignmentMaxDistance: double.NaN,
                assignmentMargin: double.NaN,
                poseWristSeparation: double.NaN,
                selectedHand0Identity: string.Empty,
                outcome: Step9CHandIdentityOutcome.Ambiguous,
                reason: reason);
            return new Step9CAmbiguousHandIdentityException(diagnostic);
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

        static void SetBridgeProperty(Step9CHandIdentityObservationBridge bridge, string name, object value)
        {
            PropertyInfo prop = typeof(Step9CHandIdentityObservationBridge).GetProperty(
                name, BindingFlags.Instance | BindingFlags.Public);
            Assert.IsNotNull(prop, "Missing bridge property: " + name);
            MethodInfo setter = prop.GetSetMethod(true);
            Assert.IsNotNull(setter, "No setter for bridge property: " + name);
            setter.Invoke(bridge, new[] { value });
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

        static void RaiseBridgeFailureWithProviderState(
            Step9CHandIdentityObservationBridge bridge,
            Exception exception,
            int providerFrameCount,
            long lastLifetimeProviderFrameIndex)
        {
            SetBridgeProperty(bridge, "ProviderFrameCount", providerFrameCount);
            SetBridgeProperty(bridge, "LastLifetimeProviderFrameIndex", lastLifetimeProviderFrameIndex);
            RaiseBridgeFailure(bridge, exception);
        }

        static int CountOccurrences(string text, string pattern)
        {
            int count = 0;
            int index = 0;
            while ((index = text.IndexOf(pattern, index, StringComparison.Ordinal)) >= 0)
            {
                count++;
                index += pattern.Length;
            }
            return count;
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
