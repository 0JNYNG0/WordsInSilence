using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Core;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.PhysicalCompatibility;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PhysicalWebcamCoordinateCompatibilityV1Tests
    {
        const double Tolerance = 1e-12;
        static string Step9BFixtureRoot => Path.Combine(Application.dataPath, "_Project/Tests/EditMode/Fixtures/LiveMediaPipeInputV1");
        static string Step9AFixtureRoot => Path.Combine(Application.dataPath, "_Project/Tests/EditMode/Fixtures/RuntimeRecognitionServiceV1");
        static string DuplicateFixtureRoot => Path.Combine(Application.dataPath, "_Project/Tests/EditMode/Fixtures/Step9CPhysicalDuplicateHandednessV1");
        static string PhysicalCaptureRoot => Path.Combine(Directory.GetParent(Application.dataPath).FullName, "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/physical_captures");
        static string TemplateDirectory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1/Templates");
        static string ThresholdPath => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json");
        const string ApprovedMapperBridgeSha256 = "2e054aeabdba0148aef9c8e016c454d138edbee5206073efbc98d2a9e5d1527a";
        const string ActualThankYouEvidenceName = "take_03_THANK_YOU_step9c-take-03-thank-you-98116ec1-005c-454a-b4de-de5a0ffcb114.json";

        [Test]
        public void RepeatedPhysicalFailureFixture_VerifiesAllReadOnlyPairsAndDocumentsLegacyObservabilityLimit()
        {
            string[] rows = File.ReadAllLines(Path.Combine(DuplicateFixtureRoot, "physical_evidence_manifest.csv")).Skip(1).Where(line => line.Length > 0).ToArray();
            Assert.AreEqual(19, rows.Length);
            Assert.AreEqual(20, Directory.GetFiles(PhysicalCaptureRoot, "*.json", SearchOption.TopDirectoryOnly).Length);
            Assert.AreEqual(20, Directory.GetFiles(PhysicalCaptureRoot, "*.json.sha256", SearchOption.TopDirectoryOnly).Length);
            int valid = 0, duplicateLeft = 0, duplicateRight = 0, diagnostic = 0;
            var before = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string row in rows)
            {
                string[] columns = row.Split(',');
                string path = Path.Combine(PhysicalCaptureRoot, columns[0]);
                string checksumPath = path + ".sha256";
                before[columns[0]] = Sha256(File.ReadAllBytes(path));
                before[columns[0] + ".sha256"] = Sha256(File.ReadAllBytes(checksumPath));
                Assert.AreEqual(long.Parse(columns[1], CultureInfo.InvariantCulture), new FileInfo(path).Length, columns[0]);
                Assert.AreEqual(columns[2], before[columns[0]], columns[0]);
                Assert.IsTrue(PhysicalCaptureEvidenceFile.Verify(path, checksumPath), columns[0]);
                string json = File.ReadAllText(path, Encoding.UTF8);
                StringAssert.Contains($"\"expected_word\":\"{columns[3]}\"", json, columns[0]);
                StringAssert.Contains($"\"raw_frames\":{columns[4]}", json, columns[0]);
                StringAssert.Contains($"\"pre_export_objectively_valid\":{columns[5].ToLowerInvariant()}", json, columns[0]);
                if (columns[6] == "DIAGNOSTIC")
                {
                    diagnostic++;
                    StringAssert.Contains("\"duplicate_frames\":4", json);
                    StringAssert.Contains("\"resolved_by_pose_wrist\":3", json);
                    StringAssert.Contains("\"ambiguous\":1", json);
                }
                else StringAssert.DoesNotContain("\"hand_identity_diagnostics\"", json, "legacy capture did not persist the rejected provider frame");
                if (columns[5] == "True") valid++;
                else if (columns[6] == "Left") { duplicateLeft++; StringAssert.Contains("duplicate Left handedness categories", json); }
                else if (columns[6] == "Right") { duplicateRight++; StringAssert.Contains("duplicate Right handedness categories", json); }
            }
            const string approvedThankYouName = "take_03_THANK_YOU_step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42.json";
            string approvedThankYouPath = Path.Combine(PhysicalCaptureRoot, approvedThankYouName);
            string approvedThankYouChecksumPath = approvedThankYouPath + ".sha256";
            before[approvedThankYouName] = Sha256(File.ReadAllBytes(approvedThankYouPath));
            before[approvedThankYouName + ".sha256"] = Sha256(File.ReadAllBytes(approvedThankYouChecksumPath));
            Assert.AreEqual(179564L, new FileInfo(approvedThankYouPath).Length, approvedThankYouName);
            Assert.AreEqual("c30e68736fd9762864c6d5da3cb5e2c995060c0136a99e48391e8f5b5572f5c2", before[approvedThankYouName], approvedThankYouName);
            Assert.IsTrue(PhysicalCaptureEvidenceFile.Verify(approvedThankYouPath, approvedThankYouChecksumPath), approvedThankYouName);
            string approvedThankYouJson = File.ReadAllText(approvedThankYouPath, Encoding.UTF8);
            StringAssert.Contains("\"expected_word\":\"THANK_YOU\"", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"raw_frames\":162", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"pre_export_objectively_valid\":true", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"finalization_state\":\"Completed\"", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"duplicate_frames\":1", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"resolved_by_pose_wrist\":1", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"ambiguous\":0", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"predicted_word\":\"THANK_YOU\"", approvedThankYouJson, approvedThankYouName);
            StringAssert.Contains("\"accepted\":false", approvedThankYouJson, approvedThankYouName);
            Assert.AreEqual(2, valid);
            Assert.AreEqual(12, duplicateLeft);
            Assert.AreEqual(4, duplicateRight);
            Assert.AreEqual(1, diagnostic);
            foreach (KeyValuePair<string, string> item in before)
                Assert.AreEqual(item.Value, Sha256(File.ReadAllBytes(Path.Combine(PhysicalCaptureRoot, item.Key))), item.Key);
            TestContext.WriteLine("LEGACY_DUPLICATE_EXACT_FRAME_TIMESTAMP_SCORE=UNAVAILABLE_NOT_PERSISTED");
        }

        [Test]
        public void DuplicateProviderCategories_OldStep9BThrowsButStep9CResolvesOnlyUnambiguousPoseWristIdentity()
        {
            HandPoseFrame source = DuplicateProviderFrame(2.5f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f);
            var approvedMapper = new MediaPipeHandPoseObservationMapper();
            InvalidOperationException oldFailure = Assert.Throws<InvalidOperationException>(() => approvedMapper.Map(source));
            StringAssert.Contains("duplicate Left handedness categories", oldFailure.Message);

            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(source, 17, out Step9CHandIdentityDiagnostic diagnostic);
            Assert.AreEqual(Step9CHandIdentityOutcome.ResolvedByPoseWrist, diagnostic.Outcome);
            Assert.AreEqual(17, diagnostic.ProviderFrameIndex);
            Assert.AreEqual("Left", diagnostic.SourceCategory0);
            Assert.AreEqual("Left", diagnostic.SourceCategory1);
            Assert.AreEqual(0.91, diagnostic.SourceScore0, 1e-6);
            Assert.AreEqual(0.86, diagnostic.SourceScore1, 1e-6);
            Assert.AreEqual("Left", diagnostic.SelectedHand0Identity);
            RawMediaPipeObservation observation = approvedMapper.Map(resolved);
            Assert.IsTrue(observation.LeftHand.IsValid(0));
            Assert.IsTrue(observation.RightHand.IsValid(0));

            HandPoseFrame ambiguous = DuplicateProviderFrame(3.0f, false, 0.49f, 0.51f, 0.40f, 0.60f, 0.77f, 0.73f);
            Step9CAmbiguousHandIdentityException ambiguity = Assert.Throws<Step9CAmbiguousHandIdentityException>(
                () => resolver.Resolve(ambiguous, 23, out _));
            Assert.AreEqual(Step9CHandIdentityOutcome.Ambiguous, ambiguity.Diagnostic.Outcome);
            Assert.AreEqual("pose_wrist_assignment_margin_ambiguous", ambiguity.Diagnostic.Reason);
            Assert.AreEqual("Right", ambiguity.Diagnostic.SourceCategory0);
            Assert.AreEqual("Right", ambiguity.Diagnostic.SourceCategory1);
        }

        [Test]
        public void DuplicateCapturePolicy_AllowsAtMostTwoResolvedTransientFramesAndRejectsAmbiguousOrStructuralRate()
        {
            var resolver = new Step9CHandIdentityResolver();
            var resolved = new List<Step9CHandIdentityDiagnostic>();
            foreach (int index in new[] { 20, 21 })
            {
                resolver.Resolve(DuplicateProviderFrame(index, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.90f, 0.85f), index, out Step9CHandIdentityDiagnostic diagnostic);
                resolved.Add(diagnostic);
            }
            CollectionAssert.IsEmpty(Step9CHandIdentityCapturePolicy.InvalidReasons(150, resolved));

            resolver.Resolve(DuplicateProviderFrame(40f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.90f, 0.85f), 40, out Step9CHandIdentityDiagnostic third);
            resolved.Add(third);
            CollectionAssert.Contains(Step9CHandIdentityCapturePolicy.InvalidReasons(150, resolved), "duplicate_handedness_not_transient");

            Step9CAmbiguousHandIdentityException ambiguity = Assert.Throws<Step9CAmbiguousHandIdentityException>(() =>
                resolver.Resolve(DuplicateProviderFrame(50f, true, 0.49f, 0.51f, 0.40f, 0.60f, 0.70f, 0.69f), 50, out _));
            CollectionAssert.Contains(Step9CHandIdentityCapturePolicy.InvalidReasons(150, new[] { ambiguity.Diagnostic }), "ambiguous_hand_identity_observed");
        }

        [Test]
        public void CaptureLifecycle_StartDiscardsPreStartAmbiguityAndInitializesCaptureOnlyState()
        {
            var ledger = new Step9CCaptureDiagnosticLedger();
            Step9CHandIdentityDiagnostic ambiguity = AmbiguousDiagnostic(9, 9.0f);
            ledger.Prepare(Boundary(10, 9, 40, 39, 8.0, 9.0, 9.0));
            Assert.IsFalse(ledger.TryRecordDiagnostic(ambiguity));
            Assert.IsFalse(ledger.TryRecordFailure(new Step9CAmbiguousHandIdentityException(ambiguity)));

            ledger.Begin(Boundary(0, 0, 40, 39, double.NaN, double.NaN, 9.0));
            Step9CHandIdentityDiagnostic resolved = ResolvedDiagnostic(0, 10.0f);
            Assert.IsTrue(ledger.TryRecordDiagnostic(resolved));
            Step9CCaptureDiagnosticSnapshot snapshot = ledger.Finalize(Boundary(1, 1, 41, 40, 10.0, 10.0, 10.0));

            Assert.AreEqual(1, snapshot.Diagnostics.Count);
            Assert.AreSame(resolved, snapshot.Diagnostics[0]);
            Assert.IsEmpty(snapshot.FailureType);
            Assert.AreEqual(0, snapshot.Window.FirstCaptureProviderFrameIndex);
            Assert.AreEqual(0, snapshot.Window.FinalCaptureProviderFrameIndex);
            Assert.AreEqual(40, snapshot.Window.FirstCaptureLifetimeProviderFrameIndex);
            Assert.AreEqual(40, snapshot.Window.FinalCaptureLifetimeProviderFrameIndex);
        }

        [Test]
        public void CaptureLifecycle_FinalizeFreezesSnapshotAndPostFinalizePreviewEventsCannotMutateIt()
        {
            var ledger = new Step9CCaptureDiagnosticLedger();
            ledger.Prepare(Boundary(3, 3, 3, 2, 1.0, 3.0, 3.0));
            ledger.Begin(Boundary(0, 0, 3, 2, double.NaN, double.NaN, 3.0));
            Step9CHandIdentityDiagnostic resolved = ResolvedDiagnostic(0, 4.0f);
            ledger.TryRecordDiagnostic(resolved);
            Step9CCaptureDiagnosticSnapshot snapshot = ledger.Finalize(Boundary(1, 1, 4, 3, 4.0, 4.0, 4.0));

            Step9CHandIdentityDiagnostic liveAmbiguity = AmbiguousDiagnostic(443, 100.9293f);
            Assert.IsFalse(ledger.TryRecordDiagnostic(liveAmbiguity));
            Assert.IsFalse(ledger.TryRecordFailure(new Step9CAmbiguousHandIdentityException(liveAmbiguity)));
            Assert.AreEqual(1, snapshot.Diagnostics.Count);
            Assert.AreSame(resolved, snapshot.Diagnostics[0]);
            Assert.IsEmpty(snapshot.FailureType);
            Assert.AreEqual(1, snapshot.Window.ProviderFrameCount);
        }

        [Test]
        public void CaptureLifecycle_InWindowAmbiguityRemainsTypedAndObjectivelyInvalid()
        {
            var ledger = new Step9CCaptureDiagnosticLedger();
            ledger.Prepare(Step9CProviderBoundary.Empty);
            ledger.Begin(Step9CProviderBoundary.Empty);
            Step9CHandIdentityDiagnostic ambiguity = AmbiguousDiagnostic(0, 1.0f);
            var failure = new Step9CAmbiguousHandIdentityException(ambiguity);
            Assert.IsTrue(ledger.TryRecordDiagnostic(ambiguity));
            Assert.IsTrue(ledger.TryRecordFailure(failure));
            Step9CCaptureDiagnosticSnapshot snapshot = ledger.Finalize(Boundary(1, 0, 1, 0, 1.0, 1.0, 1.0));

            Assert.AreEqual(typeof(Step9CAmbiguousHandIdentityException).FullName, snapshot.FailureType);
            CollectionAssert.Contains(Step9CHandIdentityCapturePolicy.InvalidReasons(1, snapshot.Diagnostics), "ambiguous_hand_identity_observed");
        }

        [Test]
        public void ActualThankYouFiveDuplicateFixture_ReconcilesFourCaptureEventsAndOnePostFinalizeLiveEvent()
        {
            string evidencePath = Path.Combine(PhysicalCaptureRoot, ActualThankYouEvidenceName);
            Assert.IsTrue(PhysicalCaptureEvidenceFile.Verify(evidencePath, evidencePath + ".sha256"));
            ActualEvidenceDto actual = JsonUtility.FromJson<ActualEvidenceDto>(File.ReadAllText(evidencePath, Encoding.UTF8));
            Assert.IsNotNull(actual);
            Assert.AreEqual(191, actual.device.provider_frame_count);
            Assert.AreEqual(190, actual.device.bridge_observation_count);
            Assert.AreEqual(4, actual.hand_identity_diagnostics.duplicate_frames);
            Assert.AreEqual(3, actual.hand_identity_diagnostics.resolved_by_pose_wrist);
            Assert.AreEqual(1, actual.hand_identity_diagnostics.ambiguous);

            string fixturePath = Path.Combine(DuplicateFixtureRoot, "thank_you_98116ec1_duplicate_events.csv");
            string[] lines = File.ReadAllLines(fixturePath).Where(line => !string.IsNullOrWhiteSpace(line)).ToArray();
            string[] header = lines[0].Split(',');
            Dictionary<string, int> columns = header.Select((name, index) => new { name, index }).ToDictionary(item => item.name, item => item.index, StringComparer.Ordinal);
            string[][] rows = lines.Skip(1).Select(line => line.Split(',')).ToArray();
            Assert.AreEqual(5, rows.Length);

            for (int index = 0; index < 4; index++)
            {
                string[] row = rows[index];
                ActualHandIdentityEventDto item = actual.hand_identity_diagnostics.events[index];
                Assert.AreEqual("CAPTURE", row[columns["scope"]]);
                Assert.AreEqual(int.Parse(row[columns["provider_frame"]], CultureInfo.InvariantCulture), item.provider_frame_index);
                Assert.That(item.provider_timestamp_seconds, Is.EqualTo(Parse(row[columns["provider_timestamp"]])).Within(Tolerance));
                Assert.That(item.hand_frame_timestamp_seconds, Is.EqualTo(Parse(row[columns["hand_timestamp"]])).Within(Tolerance));
                Assert.AreEqual(row[columns["category0"]], item.source_hands[0].category);
                Assert.AreEqual(row[columns["category1"]], item.source_hands[1].category);
                Assert.That(item.source_hands[0].score, Is.EqualTo(Parse(row[columns["score0"]])).Within(Tolerance));
                Assert.That(item.source_hands[1].score, Is.EqualTo(Parse(row[columns["score1"]])).Within(Tolerance));
                Assert.That(item.pose_wrists.left_visibility, Is.EqualTo(Parse(row[columns["pose_left_visibility"]])).Within(Tolerance));
                Assert.That(item.pose_wrists.right_visibility, Is.EqualTo(Parse(row[columns["pose_right_visibility"]])).Within(Tolerance));
                Assert.That(item.assignment.direct_cost, Is.EqualTo(Parse(row[columns["direct_cost"]])).Within(Tolerance));
                Assert.That(item.assignment.swapped_cost, Is.EqualTo(Parse(row[columns["swapped_cost"]])).Within(Tolerance));
                Assert.That(item.assignment.best_max_distance, Is.EqualTo(Parse(row[columns["best_max_distance"]])).Within(Tolerance));
                Assert.That(item.assignment.margin, Is.EqualTo(Parse(row[columns["assignment_margin"]])).Within(Tolerance));
                Assert.AreEqual(row[columns["outcome"]], item.outcome);
                Assert.AreEqual(row[columns["reason"]], item.reason);
            }

            string[] postFinalize = rows[4];
            Assert.AreEqual("POST_FINALIZE", postFinalize[columns["scope"]]);
            Assert.AreEqual("443", postFinalize[columns["provider_frame"]]);
            Assert.AreEqual("best_pose_wrist_assignment_too_far", postFinalize[columns["reason"]]);
            Assert.IsFalse(actual.hand_identity_diagnostics.events.Any(item => item.provider_frame_index == 443));

            int captureDuplicates = rows.Count(row => row[columns["scope"]] == "CAPTURE");
            int resolved = rows.Count(row => row[columns["scope"]] == "CAPTURE" && row[columns["outcome"]] == "ResolvedByPoseWrist");
            int ambiguous = captureDuplicates - resolved;
            int longestRun = LongestConsecutive(rows.Where(row => row[columns["scope"]] == "CAPTURE").Select(row => int.Parse(row[columns["provider_frame"]], CultureInfo.InvariantCulture)));
            Assert.AreEqual(4, captureDuplicates);
            Assert.AreEqual(3, resolved);
            Assert.AreEqual(1, ambiguous);
            Assert.That((double)captureDuplicates / actual.device.provider_frame_count, Is.EqualTo(4.0 / 191.0).Within(Tolerance));
            Assert.AreEqual(3, longestRun);
            Assert.Greater(resolved, Step9CHandIdentityCapturePolicy.MaximumTransientDuplicateFrames);
            Assert.Greater((double)captureDuplicates / actual.device.provider_frame_count, Step9CHandIdentityCapturePolicy.MaximumTransientDuplicateRatio);
            Assert.Greater(longestRun, Step9CHandIdentityCapturePolicy.MaximumTransientConsecutiveFrames);
        }

        [Test]
        public void NewEvidence_SerializesExactProviderFrameTimestampCategoriesScoresAndAssignmentOutcome()
        {
            var resolver = new Step9CHandIdentityResolver();
            resolver.Resolve(DuplicateProviderFrame(7.25f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f), 0, out Step9CHandIdentityDiagnostic diagnostic);
            RuntimeRecognitionService runtime = LoadService();
            PhysicalTakePlan plan = PlanForTake(3);
            PhysicalCaptureEvidence evidence = new PhysicalCaptureEvidenceRecorder(runtime, new ApprovedLivePreprocessingAdapter()).Create(
                Context(plan, 1),
                new[] { MakeSynthetic(7.25, true) },
                ManualDiagnosticCaptureState.Failed,
                null,
                "SyntheticFinalization",
                "diagnostic serialization fixture",
                new[] { diagnostic });
            CollectionAssert.IsEmpty(PhysicalCaptureEvidenceSchema.Validate(evidence));
            Assert.AreEqual(1, evidence.DuplicateHandednessFrameCount);
            Assert.AreEqual(1, evidence.ResolvedDuplicateHandednessFrameCount);
            string json = PhysicalCaptureEvidenceJson.Serialize(evidence);
            StringAssert.Contains("\"provider_frame_index\":0", json);
            StringAssert.Contains("\"provider_timestamp_seconds\":7.25", json);
            StringAssert.Contains("\"category\":\"Left\",\"score\":0.91", json);
            StringAssert.Contains("\"category\":\"Left\",\"score\":0.86", json);
            StringAssert.Contains("\"outcome\":\"ResolvedByPoseWrist\"", json);
            StringAssert.Contains("\"reason\":\"pose_wrist_assignment_unambiguous\"", json);
        }

        [Test]
        public void NewEvidence_SerializesImmutableCaptureStartAndFinalizeProviderBoundaries()
        {
            var ledger = new Step9CCaptureDiagnosticLedger();
            ledger.Prepare(Boundary(8, 8, 20, 19, 4.0, 5.0, 5.0));
            ledger.Begin(Boundary(0, 0, 20, 19, double.NaN, double.NaN, 5.25));
            Step9CCaptureDiagnosticSnapshot snapshot = ledger.Finalize(Boundary(1, 1, 21, 20, 6.0, 6.0, 6.0));
            PhysicalTakePlan plan = PlanForTake(3);
            var context = new PhysicalCaptureContext(
                plan,
                new DateTimeOffset(2026, 8, 8, 0, 0, 0, TimeSpan.Zero),
                new DateTimeOffset(2026, 8, 8, 9, 0, 0, TimeSpan.FromHours(9)),
                "6000.3.14f1",
                "0.16.3",
                "Step9C_Test",
                new PhysicalDeviceSnapshot("Synthetic Camera", 640, 480, 30, 640, 480, true, true, true, true, 1, 1),
                snapshot.Window);
            PhysicalCaptureEvidence evidence = new PhysicalCaptureEvidenceRecorder(LoadService(), new ApprovedLivePreprocessingAdapter()).Create(
                context,
                new[] { MakeSynthetic(6.0, true) },
                ManualDiagnosticCaptureState.Failed,
                null,
                "SyntheticFinalization",
                "capture window fixture",
                snapshot.Diagnostics);

            CollectionAssert.IsEmpty(PhysicalCaptureEvidenceSchema.Validate(evidence));
            string json = PhysicalCaptureEvidenceJson.Serialize(evidence);
            StringAssert.Contains("\"capture_window\":{", json);
            StringAssert.Contains("\"prepare_after_lifetime_provider_frame_index\":19", json);
            StringAssert.Contains("\"start_next_lifetime_provider_frame_index\":20", json);
            StringAssert.Contains("\"first_capture_provider_frame_index\":0", json);
            StringAssert.Contains("\"final_capture_provider_frame_index\":0", json);
            StringAssert.Contains("\"first_capture_provider_timestamp_seconds\":6", json);
            StringAssert.Contains("\"final_capture_provider_timestamp_seconds\":6", json);
        }

        [Test]
        public void EvidenceRecorder_IsImmutableAndSchemaValid()
        {
            PhysicalCaptureEvidence evidence = BuildApprovedEvidence();
            CollectionAssert.IsEmpty(PhysicalCaptureEvidenceSchema.Validate(evidence));
            Assert.IsTrue(evidence.PreExportObjectivelyValid);
            Assert.AreEqual(131, evidence.RawFrameCount);
            Assert.AreEqual(131, evidence.NormalizedFrameCount);
            Assert.AreEqual(9, evidence.Recognition.TemplateDistances.Count);

            bool[] mask = evidence.Frames[0].CopyNormalizedMask();
            double[] coordinates = evidence.Frames[0].CopyNormalizedCoordinates();
            bool originalMask = evidence.Frames[0].CopyNormalizedMask()[0];
            double originalCoordinate = evidence.Frames[0].CopyNormalizedCoordinates()[0];
            mask[0] = !mask[0];
            coordinates[0] = 999.0;
            Assert.AreEqual(originalMask, evidence.Frames[0].CopyNormalizedMask()[0]);
            Assert.AreEqual(originalCoordinate, evidence.Frames[0].CopyNormalizedCoordinates()[0]);
            foreach (PropertyInfo property in typeof(PhysicalCaptureEvidence).GetProperties(BindingFlags.Instance | BindingFlags.Public))
                Assert.IsFalse(property.CanWrite, property.Name);
        }

        [Test]
        public void Json_PreservesInvalidCoordinatesAsNullAndNeverZeroFillsInvalidLandmarks()
        {
            var observations = new[] { MakeSynthetic(1.0, includeLeftHand: false), MakeSynthetic(1.04, includeLeftHand: true) };
            PhysicalCaptureEvidence evidence = BuildFailureEvidence(observations, "SyntheticFailure", "expected test failure");
            string json = PhysicalCaptureEvidenceJson.Serialize(evidence);
            StringAssert.Contains("[null,null,null]", json);
            StringAssert.Contains("\"failure\":{\"exception_type\":\"SyntheticFailure\"", json);
            foreach (PhysicalFrameEvidence frame in evidence.Frames)
            {
                bool[] mask = frame.CopyNormalizedMask();
                double[] coordinates = frame.CopyNormalizedCoordinates();
                for (int landmark = 0; landmark < 16; landmark++)
                    if (!mask[landmark])
                        for (int axis = 0; axis < 3; axis++) Assert.IsTrue(double.IsNaN(coordinates[landmark * 3 + axis]));
            }
        }

        [Test]
        public void TimestampStatistics_AreExactAndNonMonotonicInputIsRejectedFromNormalization()
        {
            PhysicalCaptureEvidence valid = BuildFailureEvidence(new[]
            {
                MakeSynthetic(1.00, true), MakeSynthetic(1.04, true), MakeSynthetic(1.10, true),
            }, "SyntheticFailure", "statistics only");
            Assert.IsTrue(valid.TimestampsStrictlyIncreasing);
            Assert.That(valid.TimestampDeltaMin, Is.EqualTo(0.04).Within(Tolerance));
            Assert.That(valid.TimestampDeltaMedian, Is.EqualTo(0.05).Within(Tolerance));
            Assert.That(valid.TimestampDeltaMax, Is.EqualTo(0.06).Within(Tolerance));

            PhysicalCaptureEvidence duplicate = BuildFailureEvidence(new[]
            {
                MakeSynthetic(2.0, true), MakeSynthetic(2.0, true),
            }, "TimestampFailure", "duplicate timestamp");
            Assert.IsFalse(duplicate.TimestampsStrictlyIncreasing);
            Assert.AreEqual(0, duplicate.NormalizedFrameCount);
            CollectionAssert.Contains(duplicate.ObjectiveInvalidReasons, "timestamps_not_strictly_increasing");
        }

        [Test]
        public void PresenceAndBodyTransformStatistics_AreComputedFromRawObservations()
        {
            PhysicalCaptureEvidence evidence = BuildFailureEvidence(new[]
            {
                MakeSynthetic(3.00, false), MakeSynthetic(3.04, true), MakeSynthetic(3.08, true, validShoulders: false),
            }, "SyntheticFailure", "statistics only");
            Assert.AreEqual(3, evidence.PosePresentFrameCount);
            Assert.AreEqual(2, evidence.LeftHandPresentFrameCount);
            Assert.AreEqual(0, evidence.RightHandPresentFrameCount);
            Assert.AreEqual(2, evidence.BodyTransformAvailableFrameCount);
            Assert.That(evidence.ShoulderWidthMin, Is.EqualTo(2.0).Within(Tolerance));
            Assert.That(evidence.ShoulderWidthMedian, Is.EqualTo(2.0).Within(Tolerance));
            Assert.That(evidence.ShoulderWidthMax, Is.EqualTo(2.0).Within(Tolerance));
        }

        [Test]
        public void RecognitionAndTypedFailure_AreSerializedWithoutAmbiguity()
        {
            string recognitionJson = PhysicalCaptureEvidenceJson.Serialize(BuildApprovedEvidence());
            StringAssert.Contains("\"template_distances\":[", recognitionJson);
            StringAssert.Contains("\"frozen_threshold\":0.2908679008489196", recognitionJson);
            StringAssert.Contains("\"threshold_rule\":\"inclusive_less_than_or_equal\"", recognitionJson);
            StringAssert.Contains("\"failure\":null", recognitionJson);

            string failureJson = PhysicalCaptureEvidenceJson.Serialize(BuildFailureEvidence(
                new[] { MakeSynthetic(4.0, true) }, "CameraPermissionDenied", "permission denied"));
            StringAssert.Contains("\"recognition\":null", failureJson);
            StringAssert.Contains("\"exception_type\":\"CameraPermissionDenied\"", failureJson);
            StringAssert.Contains("\"message\":\"permission denied\"", failureJson);
        }

        [Test]
        public void EvidenceFile_UsesStrictDetachedSha256AndRejectsTampering()
        {
            string directory = Path.Combine(Path.GetTempPath(), "WordsInSilenceStep9C_" + Guid.NewGuid().ToString("N"));
            string path = Path.Combine(directory, "capture.json");
            try
            {
                PhysicalEvidenceExportReceipt receipt = PhysicalCaptureEvidenceFile.Export(path, BuildApprovedEvidence());
                Assert.IsTrue(receipt.HashVerified);
                Assert.IsTrue(PhysicalCaptureEvidenceFile.Verify(receipt.EvidencePath, receipt.ChecksumPath));
                File.AppendAllText(receipt.EvidencePath, "tamper");
                Assert.IsFalse(PhysicalCaptureEvidenceFile.Verify(receipt.EvidencePath, receipt.ChecksumPath));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Test]
        public void ResumeVerifier_ReadOnlyValidatesEveryPairAndRestoresFirstValidGreetingAndLove()
        {
            string directory = Path.Combine(Path.GetTempPath(), "WordsInSilenceStep9CResume_" + Guid.NewGuid().ToString("N"));
            try
            {
                PhysicalCaptureEvidence invalidGreeting = BuildFailureEvidenceForTake(1, "SyntheticBridgeFailure", "invalid first attempt");
                PhysicalCaptureEvidence validGreeting = BuildApprovedEvidenceForTake(1);
                PhysicalCaptureEvidence validLove = BuildApprovedEvidenceForTake(2);
                PhysicalCaptureEvidence invalidThankYou = BuildFailureEvidenceForTake(3, "SyntheticBridgeFailure", "invalid third take");
                ExportNamed(directory, invalidGreeting);
                ExportNamed(directory, validGreeting);
                ExportNamed(directory, validLove);
                ExportNamed(directory, invalidThankYou);

                Dictionary<string, string> before = Directory.GetFiles(directory).ToDictionary(path => Path.GetFileName(path), path => Convert.ToBase64String(File.ReadAllBytes(path)), StringComparer.Ordinal);
                PhysicalEvidenceResumeReport report = PhysicalCaptureEvidenceResume.VerifyReadOnly(directory);
                Dictionary<string, string> after = Directory.GetFiles(directory).ToDictionary(path => Path.GetFileName(path), path => Convert.ToBase64String(File.ReadAllBytes(path)), StringComparer.Ordinal);

                Assert.AreEqual(4, report.JsonFileCount);
                Assert.AreEqual(4, report.ChecksumFileCount);
                Assert.AreEqual(4, report.VerifiedCaptures.Count);
                Assert.AreEqual(2, report.RecoveredValidCaptures.Count);
                Assert.AreEqual(validGreeting.CaptureId, report.RecoveredValidCaptures[0].CaptureId);
                Assert.AreEqual(validLove.CaptureId, report.RecoveredValidCaptures[1].CaptureId);
                CollectionAssert.AreEquivalent(before, after);

                var protocol = new FixedPhysicalCaptureProtocol(Step9CPhysicalCompatibilityConstants.RunId);
                foreach (RecoveredPhysicalCapture recovered in report.RecoveredValidCaptures)
                    protocol.RestoreVerifiedExport(recovered.CaptureId, recovered.TakeNumber, recovered.ExpectedWord);
                Assert.AreEqual(2, protocol.CompletedTakeCount);
                Assert.AreEqual(3, protocol.CurrentTakeNumber);
                Assert.AreEqual("THANK_YOU", protocol.CurrentExpectedWord);
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Test]
        public void ResumeVerifier_RejectsTamperAndOrphanChecksumWithoutSelectingEvidence()
        {
            string directory = Path.Combine(Path.GetTempPath(), "WordsInSilenceStep9CResumeReject_" + Guid.NewGuid().ToString("N"));
            try
            {
                PhysicalEvidenceExportReceipt receipt = ExportNamed(directory, BuildApprovedEvidenceForTake(1));
                File.AppendAllText(receipt.EvidencePath, "tamper");
                Assert.Throws<InvalidDataException>(() => PhysicalCaptureEvidenceResume.VerifyReadOnly(directory));

                File.Delete(receipt.EvidencePath);
                Assert.Throws<InvalidDataException>(() => PhysicalCaptureEvidenceResume.VerifyReadOnly(directory));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Test]
        public void FixedProtocol_EnforcesOrderUniqueIdsAndOutcomeIndependentTakeConsumption()
        {
            var protocol = new FixedPhysicalCaptureProtocol(Step9CPhysicalCompatibilityConstants.RunId);
            PhysicalTakePlan invalidGreeting = protocol.PrepareCurrent();
            Assert.AreEqual("GREETING", invalidGreeting.ExpectedWord);
            protocol.CommitExport(invalidGreeting.CaptureId, objectivelyValid: false, evidenceHashVerified: true);
            Assert.AreEqual("GREETING", protocol.CurrentExpectedWord);

            PhysicalTakePlan greeting = protocol.PrepareCurrent();
            Assert.AreNotEqual(invalidGreeting.CaptureId, greeting.CaptureId);
            protocol.CommitExport(greeting.CaptureId, objectivelyValid: true, evidenceHashVerified: true);
            Assert.AreEqual("LOVE", protocol.CurrentExpectedWord);
            PhysicalTakePlan love = protocol.PrepareCurrent();
            protocol.CommitExport(love.CaptureId, objectivelyValid: true, evidenceHashVerified: true);
            PhysicalTakePlan thankYou = protocol.PrepareCurrent();
            Assert.AreEqual("THANK_YOU", thankYou.ExpectedWord);
            protocol.CommitExport(thankYou.CaptureId, objectivelyValid: true, evidenceHashVerified: true);
            Assert.IsTrue(protocol.IsComplete);
            Assert.Throws<InvalidOperationException>(() => protocol.PrepareCurrent());
        }

        [Test]
        public void FixedProtocol_RestoresOnlyContiguousVerifiedFixedOrder()
        {
            var protocol = new FixedPhysicalCaptureProtocol(Step9CPhysicalCompatibilityConstants.RunId);
            Assert.Throws<InvalidDataException>(() => protocol.RestoreVerifiedExport("love", 2, "LOVE"));
            protocol.RestoreVerifiedExport("greeting", 1, "GREETING");
            Assert.AreEqual("LOVE", protocol.CurrentExpectedWord);
            Assert.Throws<InvalidDataException>(() => protocol.RestoreVerifiedExport("thank-you", 3, "THANK_YOU"));
            protocol.RestoreVerifiedExport("love", 2, "LOVE");
            Assert.AreEqual("THANK_YOU", protocol.CurrentExpectedWord);
            Assert.AreEqual(3, protocol.CurrentTakeNumber);
        }

        [Test]
        public void CompatibilityClassifier_CoversProvisionalPassObservedFailAndInconclusive()
        {
            CompatibilityCaptureOutcome[] pass = Step9CPhysicalCompatibilityConstants.FixedWords
                .Select(word => new CompatibilityCaptureOutcome(word, word, true, true, true, true)).ToArray();
            Assert.AreEqual(LiveCoordinateCompatibilityClassification.ProvisionalPass, PhysicalCompatibilityClassifier.Classify(pass));

            CompatibilityCaptureOutcome[] fail = pass.ToArray();
            fail[1] = new CompatibilityCaptureOutcome("LOVE", "GREETING", true, true, true, true);
            Assert.AreEqual(LiveCoordinateCompatibilityClassification.ObservedFail, PhysicalCompatibilityClassifier.Classify(fail));
            fail[1] = new CompatibilityCaptureOutcome("LOVE", "LOVE", true, true, true, false);
            Assert.AreEqual(LiveCoordinateCompatibilityClassification.ObservedFail, PhysicalCompatibilityClassifier.Classify(fail));

            CompatibilityCaptureOutcome[] inconclusive = pass.ToArray();
            inconclusive[2] = new CompatibilityCaptureOutcome("THANK_YOU", null, false, false, false, false);
            Assert.AreEqual(LiveCoordinateCompatibilityClassification.Inconclusive, PhysicalCompatibilityClassifier.Classify(inconclusive));
        }

        [Test]
        public void CurrentApprovedMapperIdentityAndStep9BParityAndStep9ANineReplayDecisions_RemainUnchanged()
        {
            string mapperPath = Path.Combine(Application.dataPath, "_Project/Scripts/Recognition/LiveInput/MediaPipeHandPoseObservationBridge.cs");
            Assert.AreEqual(ApprovedMapperBridgeSha256, Sha256(File.ReadAllBytes(mapperPath)), "current approved mapper/bridge source identity changed");
            List<RawMediaPipeObservation> raw = LoadRawFixture();
            RecognitionSequence expectedSequence = RecognitionSequenceBinary.LoadFile(Path.Combine(Step9BFixtureRoot, "query_GREETING_NORMAL_R01.wisrq"));
            ApprovedPreprocessingResult actual = new ApprovedLivePreprocessingAdapter().Adapt(raw);
            Assert.AreEqual(131, actual.RawFrameCount);
            Assert.AreEqual(1915, actual.ValidLandmarkCount);
            Assert.AreEqual(29, actual.InterpolatedLandmarkCount);
            double maximumDelta = 0.0;
            for (int frame = 0; frame < actual.Frames.Count; frame++)
            {
                bool[] mask = actual.Frames[frame].CopyValidMask();
                double[] coordinates = actual.Frames[frame].CopyCoordinates();
                for (int landmark = 0; landmark < 16; landmark++)
                {
                    Assert.AreEqual(expectedSequence.IsValid(frame, landmark), mask[landmark]);
                    for (int axis = 0; axis < 3; axis++)
                    {
                        double wanted = expectedSequence.Coordinate(frame, landmark, axis);
                        double observed = coordinates[landmark * 3 + axis];
                        if (mask[landmark]) maximumDelta = Math.Max(maximumDelta, Math.Abs(wanted - observed));
                        else Assert.IsTrue(double.IsNaN(observed));
                    }
                }
            }
            TestContext.WriteLine($"STEP9C_STEP9B_PARITY_MAX_DELTA={maximumDelta:R}");
            Assert.LessOrEqual(maximumDelta, Tolerance);

            RuntimeRecognitionService service = LoadService();
            Dictionary<string, string[]> expected = LoadExpectedPredictions(Path.Combine(Step9AFixtureRoot, "test_expected_predictions.csv"));
            string[] paths = Directory.GetFiles(Path.Combine(Step9AFixtureRoot, "TestQueries"), "*.wisrq", SearchOption.TopDirectoryOnly);
            Assert.AreEqual(9, paths.Length);
            foreach (string path in paths)
            {
                RecognitionSequence query = RecognitionSequenceBinary.LoadFile(path);
                RuntimeRecognitionResult result = service.Recognize(query.SampleId, query);
                string[] row = expected[query.SampleId];
                Assert.AreEqual(row[3], result.PredictedWord, query.SampleId);
                Assert.AreEqual(row[5], result.WinningPrototypeId, query.SampleId);
                Assert.That(result.GreetingDistance, Is.EqualTo(Parse(row[6])).Within(Tolerance));
                Assert.That(result.LoveDistance, Is.EqualTo(Parse(row[7])).Within(Tolerance));
                Assert.That(result.ThankYouDistance, Is.EqualTo(Parse(row[8])).Within(Tolerance));
                Assert.AreEqual(bool.Parse(row[13]), result.Accepted);
            }
        }

        // =====================================================================
        //  Step 10N v1.3c2a5 — MergedGeometry preservation through resolver
        // =====================================================================

        static Step9DGeometryAcquisitionResult MakeAvailableGeometry(long tsMs = 100)
        {
            var geometry = new Step9DInferenceFrameGeometry(
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
                0L, tsMs, 1, "bottom-up");
            return Step9DGeometryAcquisitionResult.ForAvailable(geometry);
        }

        static HandPoseFrame DuplicateProviderFrameWithGeometry(float timestamp, bool duplicateLeft,
            float hand0X, float hand1X, float poseLeftX, float poseRightX, float score0, float score1)
        {
            var posePoints = new Vector3[33];
            var visibility = new float[33];
            posePoints[15] = new Vector3(poseLeftX, 0.5f, 0f);
            posePoints[16] = new Vector3(poseRightX, 0.5f, 0f);
            visibility[15] = visibility[16] = 1f;
            long tsMs = (long)(timestamp * 1000f);
            var handGeometry = MakeAvailableGeometry(tsMs);
            var poseGeometry = Step9DGeometryAcquisitionResult.ForAvailable(
                new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.ObservedAtPoseInference,
                    null, null, null, 640, 480, 0, false,
                    640, 480, 640, 480, "SRGBA",
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None, null,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    0L, tsMs, 1, "bottom-up"));
            var merged = new Step9DMergedInferenceGeometry(handGeometry, poseGeometry);
            return new HandPoseFrame
            {
                Timestamp = timestamp,
                TimestampMs = tsMs,
                HasPose = true,
                PoseFrame = new UpperBodyPoseFrame
                {
                    Timestamp = timestamp, TimestampMs = tsMs,
                    AllPoints = posePoints, Visibility = visibility,
                    InferenceGeometryResult = poseGeometry,
                },
                HandFrame = new LandmarkFrame
                {
                    Timestamp = timestamp,
                    Hands = new[]
                    {
                        Hand(duplicateLeft, hand0X, score0),
                        Hand(duplicateLeft, hand1X, score1),
                    },
                    InferenceGeometryResult = handGeometry,
                },
                MergedGeometry = merged,
            };
        }

        [Test]
        public void ResolvedDuplicate_PreservesMergedGeometry()
        {
            HandPoseFrame source = DuplicateProviderFrameWithGeometry(2.5f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f);
            Assert.IsNotNull(source.MergedGeometry, "source has MergedGeometry");
            Assert.IsTrue(source.MergedGeometry.BothAvailable, "source BothAvailable");
            Assert.IsTrue(source.MergedGeometry.FullContractCompatible, "source FullContractCompatible");

            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(source, 0, out _);

            Assert.IsNotNull(resolved.MergedGeometry, "resolved MergedGeometry must not be null");
            Assert.IsTrue(resolved.MergedGeometry.BothAvailable, "resolved BothAvailable");
            Assert.IsTrue(resolved.MergedGeometry.FullContractCompatible, "resolved FullContractCompatible");
            Assert.AreSame(source.MergedGeometry, resolved.MergedGeometry, "same MergedGeometry reference");
        }

        [Test]
        public void ResolvedDuplicate_MapperProducesNonNullMergedGeometry()
        {
            HandPoseFrame source = DuplicateProviderFrameWithGeometry(2.5f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f);
            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(source, 0, out _);

            var mapper = new MediaPipeHandPoseObservationMapper();
            RawMediaPipeObservation observation = mapper.Map(resolved);

            Assert.IsNotNull(observation.MergedGeometry, "observation MergedGeometry must not be null");
            Assert.IsTrue(observation.MergedGeometry.BothAvailable);
            Assert.IsTrue(observation.MergedGeometry.FullContractCompatible);
        }

        [Test]
        public void ResolvedDuplicate_HandFramePreservesInferenceGeometryResult()
        {
            HandPoseFrame source = DuplicateProviderFrameWithGeometry(2.5f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f);
            Assert.IsNotNull(source.HandFrame.InferenceGeometryResult);

            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(source, 0, out _);

            Assert.IsNotNull(resolved.HandFrame.InferenceGeometryResult,
                "resolved HandFrame must preserve InferenceGeometryResult");
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.Available,
                resolved.HandFrame.InferenceGeometryResult.Status);
        }

        [Test]
        public void NoDuplicate_MergedGeometryPassedThrough()
        {
            var handGeometry = MakeAvailableGeometry(100);
            var poseGeometry = Step9DGeometryAcquisitionResult.ForAvailable(
                new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.ObservedAtPoseInference,
                    null, null, null, 640, 480, 0, false,
                    640, 480, 640, 480, "SRGBA",
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None, null,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    0L, 100, 1, "bottom-up"));
            var merged = new Step9DMergedInferenceGeometry(handGeometry, poseGeometry);
            var frame = new HandPoseFrame
            {
                Timestamp = 1.0f,
                TimestampMs = 1000,
                HasPose = true,
                HandFrame = new LandmarkFrame
                {
                    Timestamp = 1.0f,
                    Hands = new[]
                    {
                        Hand(true, 0.3f, 0.95f),
                        Hand(false, 0.7f, 0.90f),
                    },
                    InferenceGeometryResult = handGeometry,
                },
                PoseFrame = new UpperBodyPoseFrame
                {
                    Timestamp = 1.0f, TimestampMs = 1000,
                    AllPoints = new Vector3[33], Visibility = new float[33],
                    InferenceGeometryResult = poseGeometry,
                },
                MergedGeometry = merged,
            };

            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(frame, 0, out Step9CHandIdentityDiagnostic diagnostic);

            Assert.IsNull(diagnostic, "no duplicate category, no diagnostic");
            Assert.AreSame(frame, resolved, "non-duplicate frame returned as-is");
            Assert.IsNotNull(resolved.MergedGeometry);
            Assert.IsTrue(resolved.MergedGeometry.BothAvailable);
        }

        [Test]
        public void MissingGeometry_NullPreservedWithoutFabrication()
        {
            HandPoseFrame source = DuplicateProviderFrame(2.5f, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f);
            Assert.IsNull(source.MergedGeometry, "DuplicateProviderFrame has no geometry");
            Assert.IsNull(source.HandFrame.InferenceGeometryResult, "no typed hand result");

            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(source, 0, out _);
            Assert.IsNull(resolved.MergedGeometry, "null geometry is preserved as null, not fabricated");
            Assert.IsNull(resolved.HandFrame.InferenceGeometryResult, "null typed hand result preserved as null");
        }

        [Test]
        public void MissingGeometry_UnavailableTypedResultPreservedThroughCloneFrame()
        {
            long tsMs = 2500;
            var unavailableHand = Step9DGeometryAcquisitionResult.ForUnavailable(
                Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady, "HAND_NOT_READY");
            var unavailablePose = Step9DGeometryAcquisitionResult.ForUnavailable(
                Step9DGeometryAcquisitionStatus.UnavailableCaptureException, "POSE_CAPTURE_ERR");
            var merged = new Step9DMergedInferenceGeometry(unavailableHand, unavailablePose);

            var posePoints = new Vector3[33];
            var visibility = new float[33];
            posePoints[15] = new Vector3(0.20f, 0.5f, 0f);
            posePoints[16] = new Vector3(0.80f, 0.5f, 0f);
            visibility[15] = visibility[16] = 1f;

            var source = new HandPoseFrame
            {
                Timestamp = 2.5f,
                TimestampMs = tsMs,
                HasPose = true,
                PoseFrame = new UpperBodyPoseFrame
                {
                    Timestamp = 2.5f, TimestampMs = tsMs,
                    AllPoints = posePoints, Visibility = visibility,
                    InferenceGeometryResult = unavailablePose,
                },
                HandFrame = new LandmarkFrame
                {
                    Timestamp = 2.5f,
                    Hands = new[]
                    {
                        Hand(true, 0.20f, 0.91f),
                        Hand(true, 0.80f, 0.86f),
                    },
                    InferenceGeometryResult = unavailableHand,
                },
                MergedGeometry = merged,
            };

            Assert.IsFalse(source.MergedGeometry.BothAvailable, "unavailable merged is not BothAvailable");
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady,
                source.HandFrame.InferenceGeometryResult.Status);
            Assert.AreEqual("HAND_NOT_READY", source.HandFrame.InferenceGeometryResult.DiagnosticCode);

            var resolver = new Step9CHandIdentityResolver();
            HandPoseFrame resolved = resolver.Resolve(source, 0, out _);

            Assert.AreSame(source.MergedGeometry, resolved.MergedGeometry,
                "same MergedGeometry reference preserved through CloneFrame");
            Assert.AreEqual(Step9DGeometryAcquisitionStatus.UnavailableBeforeFrameReady,
                resolved.HandFrame.InferenceGeometryResult.Status,
                "typed unavailable status preserved");
            Assert.AreEqual("HAND_NOT_READY",
                resolved.HandFrame.InferenceGeometryResult.DiagnosticCode,
                "typed unavailable diagnostic code preserved");
            Assert.AreSame(source.HandFrame.InferenceGeometryResult,
                resolved.HandFrame.InferenceGeometryResult,
                "same InferenceGeometryResult reference");
        }

        [Test]
        public void LegacyRecognitionDecisions_UnchangedAfterGeometryFix()
        {
            RuntimeRecognitionService service = LoadService();
            string[] paths = Directory.GetFiles(Path.Combine(Step9AFixtureRoot, "TestQueries"), "*.wisrq", SearchOption.TopDirectoryOnly);
            Assert.AreEqual(9, paths.Length);
            Dictionary<string, string[]> expected = LoadExpectedPredictions(Path.Combine(Step9AFixtureRoot, "test_expected_predictions.csv"));
            foreach (string path in paths)
            {
                RecognitionSequence query = RecognitionSequenceBinary.LoadFile(path);
                RuntimeRecognitionResult result = service.Recognize(query.SampleId, query);
                Assert.AreEqual(expected[query.SampleId][3], result.PredictedWord, query.SampleId);
                Assert.AreEqual(expected[query.SampleId][5], result.WinningPrototypeId, query.SampleId);
                Assert.AreEqual(bool.Parse(expected[query.SampleId][13]), result.Accepted, query.SampleId);
            }
        }

        static PhysicalCaptureEvidence BuildApprovedEvidence()
            => BuildApprovedEvidenceForTake(1);

        static PhysicalCaptureEvidence BuildApprovedEvidenceForTake(int takeNumber)
        {
            RuntimeRecognitionService service = LoadService();
            List<RawMediaPipeObservation> raw = LoadRawFixture();
            PhysicalTakePlan plan = PlanForTake(takeNumber);
            var controller = new ManualDiagnosticCaptureController(service, new ApprovedLivePreprocessingAdapter());
            controller.Begin(plan.CaptureId, RawObservationConvention.ApprovedStep7B);
            foreach (RawMediaPipeObservation observation in raw) controller.AddObservation(plan.CaptureId, observation);
            DiagnosticRecognitionResult result = controller.FinalizeCapture(plan.CaptureId);
            var context = Context(plan, raw.Count);
            return new PhysicalCaptureEvidenceRecorder(service, new ApprovedLivePreprocessingAdapter()).Create(
                context, raw, controller.State, result, null, null);
        }

        static PhysicalCaptureEvidence BuildFailureEvidence(IReadOnlyList<RawMediaPipeObservation> observations, string type, string message)
        {
            RuntimeRecognitionService service = LoadService();
            PhysicalTakePlan plan = PlanForTake(1);
            return new PhysicalCaptureEvidenceRecorder(service, new ApprovedLivePreprocessingAdapter()).Create(
                Context(plan, observations.Count), observations, ManualDiagnosticCaptureState.Failed, null, type, message);
        }

        static PhysicalCaptureEvidence BuildFailureEvidenceForTake(int takeNumber, string type, string message)
        {
            RuntimeRecognitionService service = LoadService();
            PhysicalTakePlan plan = PlanForTake(takeNumber);
            IReadOnlyList<RawMediaPipeObservation> observations = new[] { MakeSynthetic(4.0 + takeNumber, true) };
            return new PhysicalCaptureEvidenceRecorder(service, new ApprovedLivePreprocessingAdapter()).Create(
                Context(plan, observations.Count), observations, ManualDiagnosticCaptureState.Failed, null, type, message);
        }

        static PhysicalTakePlan PlanForTake(int takeNumber)
        {
            var protocol = new FixedPhysicalCaptureProtocol(Step9CPhysicalCompatibilityConstants.RunId);
            for (int completed = 1; completed < takeNumber; completed++)
            {
                PhysicalTakePlan prior = protocol.PrepareCurrent();
                protocol.CommitExport(prior.CaptureId, objectivelyValid: true, evidenceHashVerified: true);
            }
            return protocol.PrepareCurrent();
        }

        static PhysicalEvidenceExportReceipt ExportNamed(string directory, PhysicalCaptureEvidence evidence)
        {
            string fileName = $"take_{evidence.TakeNumber:D2}_{evidence.ExpectedWord}_{evidence.CaptureId}.json";
            return PhysicalCaptureEvidenceFile.Export(Path.Combine(directory, fileName), evidence);
        }

        static PhysicalCaptureContext Context(PhysicalTakePlan plan, int frames) => new PhysicalCaptureContext(
            plan,
            new DateTimeOffset(2026, 8, 8, 0, 0, 0, TimeSpan.Zero),
            new DateTimeOffset(2026, 8, 8, 9, 0, 0, TimeSpan.FromHours(9)),
            "6000.3.14f1",
            "0.16.3",
            "Step9C_Test",
            new PhysicalDeviceSnapshot("Synthetic Camera", 640, 480, 30, 640, 480, true, true, true, true, frames));

        static RawMediaPipeObservation MakeSynthetic(double timestamp, bool includeLeftHand, bool validShoulders = true)
        {
            double[] poseCoordinates = NaNCoordinates(33);
            bool[] poseValid = new bool[33];
            if (validShoulders)
            {
                WriteValid(poseCoordinates, poseValid, 11, -1.0, 0.0, 0.0);
                WriteValid(poseCoordinates, poseValid, 12, 1.0, 0.0, 0.0);
            }
            WriteValid(poseCoordinates, poseValid, 23, -1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 24, 1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 13, 0.0, -1.0, 0.0);
            double[] leftCoordinates = NaNCoordinates(21);
            bool[] leftValid = new bool[21];
            if (includeLeftHand) WriteValid(leftCoordinates, leftValid, 0, -0.5, -1.0, 0.0);
            return new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.LiveMediaPipe,
                new RawLandmarkObservationSet(33, poseCoordinates, poseValid, new bool[33], new bool[33]),
                new RawLandmarkObservationSet(21, leftCoordinates, leftValid, new bool[21], new bool[21]),
                RawLandmarkObservationSet.Absent(21));
        }

        static RuntimeRecognitionService LoadService() => RuntimeRecognitionService.Load(TemplateDirectory, ThresholdPath);

        static List<RawMediaPipeObservation> LoadRawFixture()
        {
            using (var stream = File.OpenRead(Path.Combine(Step9BFixtureRoot, "approved_GREETING_NORMAL_R01.raw9b")))
            using (var reader = new BinaryReader(stream))
            {
                Assert.AreEqual("WISRAW9B", new string(reader.ReadChars(8)));
                Assert.AreEqual(1, reader.ReadInt32());
                int frameCount = reader.ReadInt32();
                var frames = new List<RawMediaPipeObservation>(frameCount);
                for (int frame = 0; frame < frameCount; frame++)
                {
                    double timestamp = reader.ReadDouble();
                    frames.Add(new RawMediaPipeObservation(
                        timestamp,
                        RawObservationConvention.ApprovedStep7B,
                        ReadSet(reader, 33),
                        ReadSet(reader, 21),
                        ReadSet(reader, 21)));
                }
                Assert.AreEqual(stream.Length, stream.Position);
                return frames;
            }
        }

        static RawLandmarkObservationSet ReadSet(BinaryReader reader, int count)
        {
            var coordinates = new double[count * 3];
            var valid = new bool[count];
            var reappearance = new bool[count];
            var strong = new bool[count];
            for (int landmark = 0; landmark < count; landmark++)
            {
                int offset = landmark * 3;
                coordinates[offset] = reader.ReadDouble();
                coordinates[offset + 1] = reader.ReadDouble();
                coordinates[offset + 2] = reader.ReadDouble();
                byte flags = reader.ReadByte();
                valid[landmark] = (flags & 1) != 0;
                reappearance[landmark] = (flags & 2) != 0;
                strong[landmark] = (flags & 4) != 0;
            }
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, strong);
        }

        static Dictionary<string, string[]> LoadExpectedPredictions(string path) => File.ReadAllLines(path)
            .Skip(1).Where(line => !string.IsNullOrWhiteSpace(line)).Select(line => line.Split(','))
            .ToDictionary(row => row[0], StringComparer.Ordinal);
        static double Parse(string value) => double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
        static Step9CProviderBoundary Boundary(int provider, int observations, long lifetime, long lastLifetime, double firstTimestamp, double lastTimestamp, double lifetimeTimestamp)
            => new Step9CProviderBoundary(provider, observations, lifetime, lastLifetime, firstTimestamp, lastTimestamp, lifetimeTimestamp);
        static Step9CHandIdentityDiagnostic ResolvedDiagnostic(int frameIndex, float timestamp)
        {
            var resolver = new Step9CHandIdentityResolver();
            resolver.Resolve(DuplicateProviderFrame(timestamp, true, 0.20f, 0.80f, 0.20f, 0.80f, 0.91f, 0.86f), frameIndex, out Step9CHandIdentityDiagnostic diagnostic);
            return diagnostic;
        }
        static Step9CHandIdentityDiagnostic AmbiguousDiagnostic(int frameIndex, float timestamp)
        {
            var resolver = new Step9CHandIdentityResolver();
            return Assert.Throws<Step9CAmbiguousHandIdentityException>(() =>
                resolver.Resolve(DuplicateProviderFrame(timestamp, false, 0.49f, 0.51f, 0.40f, 0.60f, 0.77f, 0.73f), frameIndex, out _)).Diagnostic;
        }
        static int LongestConsecutive(IEnumerable<int> values)
        {
            int longest = 0, current = 0, previous = int.MinValue;
            foreach (int value in values.OrderBy(item => item))
            {
                current = value == previous + 1 ? current + 1 : 1;
                longest = Math.Max(longest, current);
                previous = value;
            }
            return longest;
        }
        static HandPoseFrame DuplicateProviderFrame(float timestamp, bool duplicateLeft, float hand0X, float hand1X, float poseLeftX, float poseRightX, float score0, float score1)
        {
            var posePoints = new Vector3[33];
            var visibility = new float[33];
            posePoints[15] = new Vector3(poseLeftX, 0.5f, 0f);
            posePoints[16] = new Vector3(poseRightX, 0.5f, 0f);
            visibility[15] = visibility[16] = 1f;
            return new HandPoseFrame
            {
                Timestamp = timestamp,
                TimestampMs = (long)(timestamp * 1000f),
                HasPose = true,
                PoseFrame = new UpperBodyPoseFrame { Timestamp = timestamp, TimestampMs = (long)(timestamp * 1000f), AllPoints = posePoints, Visibility = visibility },
                HandFrame = new LandmarkFrame
                {
                    Timestamp = timestamp,
                    Hands = new[]
                    {
                        Hand(duplicateLeft, hand0X, score0),
                        Hand(duplicateLeft, hand1X, score1),
                    },
                },
            };
        }

        static HandLandmarks Hand(bool isLeft, float wristX, float score)
        {
            var points = new LandmarkPoint[21];
            points[0] = new LandmarkPoint { X = wristX, Y = 0.5f, Z = 0f };
            return new HandLandmarks { IsLeftHand = isLeft, Confidence = score, Points = points };
        }

        static string Sha256(byte[] bytes)
        {
            using (var algorithm = SHA256.Create())
            {
                byte[] hash = algorithm.ComputeHash(bytes);
                var output = new StringBuilder(hash.Length * 2);
                foreach (byte value in hash) output.Append(value.ToString("x2", CultureInfo.InvariantCulture));
                return output.ToString();
            }
        }

        static double[] NaNCoordinates(int count) { var values = new double[count * 3]; for (int index = 0; index < values.Length; index++) values[index] = double.NaN; return values; }
        static void WriteValid(double[] coordinates, bool[] valid, int landmark, double x, double y, double z) { int offset = landmark * 3; coordinates[offset] = x; coordinates[offset + 1] = y; coordinates[offset + 2] = z; valid[landmark] = true; }

        [Serializable]
        sealed class ActualEvidenceDto
        {
            public ActualDeviceDto device;
            public ActualHandIdentityDiagnosticsDto hand_identity_diagnostics;
        }

        [Serializable]
        sealed class ActualDeviceDto
        {
            public int provider_frame_count;
            public int bridge_observation_count;
        }

        [Serializable]
        sealed class ActualHandIdentityDiagnosticsDto
        {
            public int duplicate_frames;
            public int resolved_by_pose_wrist;
            public int ambiguous;
            public ActualHandIdentityEventDto[] events;
        }

        [Serializable]
        sealed class ActualHandIdentityEventDto
        {
            public int provider_frame_index;
            public double provider_timestamp_seconds;
            public double hand_frame_timestamp_seconds;
            public ActualSourceHandDto[] source_hands;
            public ActualPoseWristsDto pose_wrists;
            public ActualAssignmentDto assignment;
            public string outcome;
            public string reason;
        }

        [Serializable]
        sealed class ActualSourceHandDto
        {
            public string category;
            public double score;
        }

        [Serializable]
        sealed class ActualPoseWristsDto
        {
            public double left_visibility;
            public double right_visibility;
        }

        [Serializable]
        sealed class ActualAssignmentDto
        {
            public double direct_cost;
            public double swapped_cost;
            public double best_max_distance;
            public double margin;
        }
    }
}
