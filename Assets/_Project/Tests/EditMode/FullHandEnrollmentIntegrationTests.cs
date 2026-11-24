// F28 EditMode tests for the full-hand practice-record collection, its store, the real
// production world gate, the honest learner copy and the actual-vocabulary world snapshot.
//
// Everything is synthetic and local: an explicit in-memory F20 common bank, explicit
// operator-style gates written by the test, raw capture geometry built by the same helpers
// the existing F20/F21/F27 tests use, and a temporary directory this test creates and
// deletes. There is no webcam, no scene, no private model, no real participant state, no
// Application persistence and no network.
//
// LIMITS, stated honestly: these tests do NOT exercise a camera, a scene transition, a real
// participant reset or the in-world MonoBehaviour construction, and they cannot create
// filesystem junctions portably, so the reparse-point guards are source-reviewed rather than
// executed here. Nothing here asserts recognition accuracy, human performance,
// personalization benefit or any probability of correctness: "Accepted" is asserted as a
// bank/gate decision only, and a stored record as a MODEL_GATE_ACCEPTED research candidate.
// NOTHING IN THIS FILE HAS BEEN EXECUTED; the operator runs it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using WordsInSilence.Practice.FullHandEnrollment;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandEnrollmentIntegrationTests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;
        const int SourceFrames = 4;
        const string Participant = "participant_f28";
        const string OtherParticipant = "participant_f28_other";
        const string Greeting = "GREETING";
        const string ThankYou = "THANK_YOU";
        const string ModelId = "model_f28_test_v1";
        const string GateConfigId = "config_accept_f28";
        const string GateAuthorityId = "gate_f28";

        string root;
        int probeCounter;
        readonly HashSet<string> filledSourceHashes = new HashSet<string>(StringComparer.Ordinal);
        readonly HashSet<string> filledFeatureDigests = new HashSet<string>(StringComparer.Ordinal);

        [SetUp]
        public void SetUp()
        {
            root = Path.Combine(Path.GetTempPath(), "wis_f28_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            probeCounter = 0;
            filledSourceHashes.Clear();
            filledFeatureDigests.Clear();
        }

        [TearDown]
        public void TearDown()
        {
            FullHandEnrollmentRuntime.ClearTestOverride();
            try { if (Directory.Exists(root)) Directory.Delete(root, true); }
            catch (Exception) { /* the operator's temp folder is not a test assertion */ }
        }

        // ---- synthetic raw capture geometry (the existing F21/F27 helpers) ------

        static Step9DInferenceFrameGeometry Frame()
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                Width, Height, 30, Width, Height, 0, false, Width, Height, Width, Height,
                "RGBA32",
                Step9DPreInferenceTransformState.None, Step9DPreInferenceTransformState.None,
                0, Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None, 0L, 0L, 0, "TopDown");
        }

        static Step9DMergedInferenceGeometry Standard()
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(Frame()),
                Step9DGeometryAcquisitionResult.ForAvailable(Frame()));
        }

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
            Put(c, v, 11, 0.35 + drift, 0.35);
            Put(c, v, 12, 0.65 + drift, 0.35);
            Put(c, v, 23, 0.40 + drift, 0.68);
            Put(c, v, 24, 0.60 + drift, 0.68);
            Put(c, v, 13, 0.30 + drift, 0.45);
            Put(c, v, 14, 0.70 + drift, 0.45);
            Put(c, v, 15, 0.28 + drift, 0.55);
            Put(c, v, 16, 0.72 + drift, 0.55);
            return new RawLandmarkObservationSet(Pose, c, v, new bool[Pose], new bool[Pose]);
        }

        static RawLandmarkObservationSet HandSet(int frame, double baseX, double baseY)
        {
            var c = NaNArray(Hand);
            var v = new bool[Hand];
            for (int joint = 0; joint < Hand; joint++)
                Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame,
                                 baseY + 0.003 * joint - 0.0005 * frame);
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], new bool[Hand]);
        }

        static List<RawMediaPipeObservation> RawCapture(int frames)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame), HandSet(frame, 0.30, 0.50),
                    HandSet(frame, 0.60, 0.50), Standard()));
            return list;
        }

        // ---- explicit synthetic F20 material -----------------------------------

        static FullHandFeatureSequence Sequence(double seed)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            var coordinates = new double[SourceFrames * points * dims];
            var mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = seed + point * 0.13 + frame * 0.05;
                    coordinates[slot * dims + 1] = seed * 0.5 + point * 0.07 - frame * 0.03;
                    coordinates[slot * dims + 2] = 0.0;
                }
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, SourceFrames);
        }

        static FullHandModel Model(int width)
        {
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            for (int index = 0; index < width; index++)
            {
                mean[index] = 0.1 * index;
                scale[index] = 1.0 + 0.01 * index;
                coefficients[index] = ((index % 3) - 1) * 0.25;
            }
            return new FullHandModel(mean, scale, coefficients, 0.2);
        }

        /// <summary>
        /// The SAME synthetic bank, with the two vocabulary NAMES in the given order. Swapping
        /// the order is a fixture device that lets the test obtain a genuinely accepted record
        /// for each of the two supported words; it says nothing about which word is correct.
        /// </summary>
        static FullHandInMemoryCommonBank Bank(string[] vocabularyOrder)
        {
            return new FullHandInMemoryCommonBank(ModelId, vocabularyOrder,
                new[] { "common_0", "common_1", "common_2", "common_3" },
                new[] { 0, 0, 1, 1 },
                new[] { Sequence(1.0), Sequence(1.4), Sequence(3.0), Sequence(3.4) },
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
        }

        /// <summary>Arbitrary permissive test literals. No production cutoff appears here.</summary>
        static FullHandGateConfig AcceptAllGates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.AtQualityThreshold(
                    GateAuthorityId, FullHandMethods.All[index], -1.0e9);
            return new FullHandGateConfig(GateConfigId, gates);
        }

        static FullHandGateConfig RejectAllGates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.RejectEverything(
                    "gate_f28_reject", FullHandMethods.All[index]);
            return new FullHandGateConfig("config_reject_f28", gates);
        }

        /// <summary>Retains the source bytes, exactly as the enabled gameplay route does.</summary>
        static FullHandGameCaptureProcessor Processor(string[] vocabularyOrder,
                                                      FullHandGateConfig gates)
        {
            var pipeline = new FullHandRecognitionPipeline(Bank(vocabularyOrder), gates,
                Participant, FullHandParticipantReferenceSet.Empty(Participant));
            return new FullHandGameCaptureProcessor(pipeline, () => Participant, true);
        }

        static FullHandGameCaptureProcessor Direct()
        {
            return Processor(new[] { Greeting, ThankYou }, AcceptAllGates());
        }

        static FullHandGameCaptureProcessor Swapped()
        {
            return Processor(new[] { ThankYou, Greeting }, AcceptAllGates());
        }

        string ProbeTop1(FullHandGameCaptureProcessor processor, int frames)
        {
            FullHandGameCaptureResult probe = processor.Process(
                "capture_f28_probe_" + (++probeCounter), null, RawCapture(frames));
            Assert.IsTrue(probe.Succeeded, probe.FailureCode + " " + probe.FailureDetail);
            string top1 = probe.Outcome.Before.Decision(FullHandMethods.Hybrid).Top1;
            Assert.IsFalse(string.IsNullOrEmpty(top1), "the probe capture must rank a class");
            return top1;
        }

        sealed class Accepted
        {
            public FullHandGameCaptureResult Evidence;
            public FullHandGameplayDecision Decision;
            public string Word;
            public string CaptureId;
        }

        /// <summary>One genuinely accepted capture: gate ACCEPTED and Top1 == declared target.</summary>
        Accepted AcceptedCapture(FullHandGameCaptureProcessor processor, int frames, string id)
        {
            string word = ProbeTop1(processor, frames);
            FullHandGameCaptureResult evidence = processor.Process(id, word, RawCapture(frames));
            Assert.IsTrue(evidence.Succeeded, evidence.FailureCode + " " + evidence.FailureDetail);
            Assert.IsTrue(evidence.HasSourceBinary,
                "the enabled gameplay route must retain the source bytes it already built");
            FullHandGameplayDecision decision =
                FullHandGameplayEvaluator.Decide(evidence, word, id, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.Accepted, decision.Kind,
                "this fixture needs an accepted, matching decision to store");
            Assert.AreEqual(GateAuthorityId, decision.GateId,
                "the fixture binding pins this synthetic authority gate id");
            return new Accepted
            {
                Evidence = evidence, Decision = decision, Word = word, CaptureId = id,
            };
        }

        static FullHandEnrollmentBinding Binding(params string[] vocabulary)
        {
            return new FullHandEnrollmentBinding(Participant, ModelId, GateConfigId, vocabulary,
                                                 GateAuthorityId);
        }

        FullHandParticipantCandidateStore Store(FullHandEnrollmentBinding binding)
        {
            return FullHandParticipantCandidateStore.Open(root, Participant, binding);
        }

        string RecordDirectory()
        {
            return Path.Combine(Path.Combine(root, Participant),
                                FullHandParticipantCandidateStore.DirectoryName);
        }

        static string OtherWord(string word)
        {
            return string.Equals(word, Greeting, StringComparison.Ordinal) ? ThankYou : Greeting;
        }

        // ---- storage: save, integrity, reload -----------------------------------

        [Test]
        public void AnAcceptedCaptureIsStoredWithItsSourceAndReloadsIntact()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_a");
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            FullHandParticipantCandidateStore store = Store(binding);
            Assert.IsTrue(store.InventoryComplete);
            Assert.AreEqual(0, store.TotalCount);

            FullHandEnrollmentSaveResult save = store.TrySave(
                accepted.Decision, accepted.Evidence, accepted.Word, DateTime.UtcNow);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved, save.Status,
                save.FailureCode + " " + save.FailureDetail);
            Assert.AreEqual(1, save.CountForWord);
            Assert.IsFalse(store.IsCollectedFor(accepted.Word),
                "one record is not the collection quorum");

            IReadOnlyList<FullHandParticipantCandidateRecord> records =
                store.RecordsFor(accepted.Word);
            Assert.AreEqual(1, records.Count);
            Assert.AreEqual(accepted.Evidence.SourceSha256, records[0].SourceSha256);
            Assert.AreEqual(accepted.Decision.QueryFeatureDigest, records[0].FeatureDigest);
            Assert.AreEqual(accepted.CaptureId, records[0].CaptureSessionId);
            Assert.AreEqual(ModelId, records[0].ModelId);
            Assert.AreEqual(GateConfigId, records[0].GateConfigId);
            Assert.AreEqual(GateAuthorityId, records[0].GateId);
            // the ONLY quality claim: a model/gate decision, never user or expert verified
            Assert.AreEqual(FullHandParticipantCandidateStore.QualityStatusModelGateAccepted,
                            records[0].QualityStatus);
            Assert.AreEqual(FullHandParticipantCandidateStore.QualityBasisLiteral,
                            records[0].QualityBasis);

            FullHandParticipantCandidateStore reopened = Store(binding);
            Assert.AreEqual(1, reopened.CountFor(accepted.Word));
            Assert.AreEqual(0, reopened.RefusedRecordCount);
            Assert.IsTrue(reopened.InventoryComplete);
        }

        [Test]
        public void TheSameCaptureEvidenceNeverCountsTwice()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_dup");
            FullHandParticipantCandidateStore store = Store(Binding(Greeting, ThankYou));

            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                store.TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                              DateTime.UtcNow).Status);
            FullHandEnrollmentSaveResult again = store.TrySave(
                accepted.Decision, accepted.Evidence, accepted.Word, DateTime.UtcNow);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Duplicate, again.Status);
            Assert.AreEqual(1, again.CountForWord);
            Assert.AreEqual(1, store.TotalCount);
        }

        [Test]
        public void ARejectedAbstentionAnUnsupportedWordAndARefusalStoreNothing()
        {
            FullHandParticipantCandidateStore store = Store(Binding(Greeting, ThankYou));

            FullHandGameCaptureProcessor abstaining =
                Processor(new[] { Greeting, ThankYou }, RejectAllGates());
            const string abstainId = "capture_f28_abstain";
            FullHandGameCaptureResult abstained =
                abstaining.Process(abstainId, Greeting, RawCapture(6));
            FullHandGameplayDecision abstention =
                FullHandGameplayEvaluator.Decide(abstained, Greeting, abstainId, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.NotDistinguished, abstention.Kind);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.NotAccepted,
                store.TrySave(abstention, abstained, Greeting, DateTime.UtcNow).Status);

            // LOVE is genuinely outside this vocabulary, exactly as it is outside the real one
            const string loveId = "capture_f28_love";
            FullHandGameCaptureResult loveEvidence =
                Direct().Process(loveId, "LOVE", RawCapture(6));
            FullHandGameplayDecision love =
                FullHandGameplayEvaluator.Decide(loveEvidence, "LOVE", loveId, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.UnsupportedWord, love.Kind);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.NotAccepted,
                store.TrySave(love, loveEvidence, "LOVE", DateTime.UtcNow).Status);

            FullHandGameplayDecision refusal = FullHandGameplayEvaluator.Decide(
                null, Greeting, "capture_f28_none", false);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Refused,
                store.TrySave(refusal, null, Greeting, DateTime.UtcNow).Status);

            Assert.AreEqual(0, store.TotalCount);
            Assert.AreEqual(0, Directory.GetFiles(RecordDirectory()).Length,
                "a refused attempt writes no file at all");
        }

        [Test]
        public void AnAcceptedCaptureForADifferentWordIsNeverStoredForThePromptedWord()
        {
            FullHandGameCaptureProcessor processor = Direct();
            string other = OtherWord(ProbeTop1(processor, 6));
            const string id = "capture_f28_wrongword";
            FullHandGameCaptureResult evidence = processor.Process(id, other, RawCapture(6));
            FullHandGameplayDecision decision =
                FullHandGameplayEvaluator.Decide(evidence, other, id, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.AcceptedDifferentWord, decision.Kind);

            FullHandParticipantCandidateStore store = Store(Binding(Greeting, ThankYou));
            Assert.AreEqual(FullHandEnrollmentSaveStatus.NotAccepted,
                store.TrySave(decision, evidence, other, DateTime.UtcNow).Status);
            Assert.AreEqual(0, store.TotalCount);
        }

        [Test]
        public void ATamperedOrTruncatedSourceBinaryIsRefusedOnReload()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_tamper");
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                Store(binding).TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                                       DateTime.UtcNow).Status);

            string[] binaries = Directory.GetFiles(RecordDirectory(), "*.bin");
            Assert.AreEqual(1, binaries.Length);
            byte[] bytes = File.ReadAllBytes(binaries[0]);
            bytes[bytes.Length - 1] ^= 0xFF;
            File.WriteAllBytes(binaries[0], bytes);

            FullHandParticipantCandidateStore tampered = Store(binding);
            Assert.AreEqual(0, tampered.CountFor(accepted.Word),
                "a changed capture is not the recorded capture and never counts");
            Assert.AreEqual(1, tampered.RefusedRecordCount);
            Assert.IsTrue(tampered.InventoryComplete,
                "one refused record does not make the scan incomplete");

            File.WriteAllBytes(binaries[0], new byte[] { 1, 2, 3, 4 });
            Assert.AreEqual(0, Store(binding).TotalCount);
        }

        [Test]
        public void AMalformedOversizedDuplicateKeyOrMissingKeyDocumentIsRefused()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_document");
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                Store(binding).TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                                       DateTime.UtcNow).Status);
            string[] documents = Directory.GetFiles(RecordDirectory(), "*.json");
            Assert.AreEqual(1, documents.Length);
            // the record is written as UTF-8 WITHOUT a BOM; the test reads and restores the
            // exact bytes and writes every corruption without a BOM too, so each refusal below
            // is caused by the corruption itself and not by an added byte-order mark
            var noBom = new UTF8Encoding(false);
            byte[] originalBytes = File.ReadAllBytes(documents[0]);
            string original = noBom.GetString(originalBytes);
            Assert.AreEqual(1, Store(binding).TotalCount, "the untouched record verifies");

            foreach (string corrupt in new[]
            {
                "{ not json at all",
                new string('x', 64 * 1024),
                // a duplicated required key
                original.Replace("{", "{\"word\":\"" + accepted.Word + "\","),
                // a removed required key
                original.Replace(",\"quality_basis\":\"" +
                    FullHandParticipantCandidateStore.QualityBasisLiteral + "\"", string.Empty),
                // an unknown extra key
                original.Replace("{", "{\"extra\":\"1\","),
                // a relabelled word no longer matches the derived record identity
                original.Replace("\"word\":\"" + accepted.Word + "\"",
                                 "\"word\":\"" + OtherWord(accepted.Word) + "\""),
                // a record id that is not the file stem
                original.Replace("\"record_id\":\"fhc_", "\"record_id\":\"fhx_"),
                // the scored sample is not this capture
                original.Replace("\"query_sample_id\":\"" + accepted.CaptureId + "\"",
                                 "\"query_sample_id\":\"other_capture\""),
                // an unstated or malformed quality basis
                original.Replace(FullHandParticipantCandidateStore.QualityBasisLiteral,
                                 "USER_VERIFIED"),
                // a malformed timestamp
                original.Replace("\"saved_utc\":\"", "\"saved_utc\":\"not-a-time"),
            })
            {
                string label = corrupt.Substring(0, Math.Min(48, corrupt.Length));
                byte[] corruptBytes = noBom.GetBytes(corrupt);
                CollectionAssert.AreNotEqual(originalBytes, corruptBytes,
                    "each corruption must actually change the stored document: " + label);
                File.WriteAllBytes(documents[0], corruptBytes);
                Assert.AreEqual(0, Store(binding).TotalCount,
                    "a corrupt or incomplete record must never count: " + label);
            }

            File.WriteAllBytes(documents[0], originalBytes);
            CollectionAssert.AreEqual(originalBytes, File.ReadAllBytes(documents[0]),
                "the original record bytes must be restored exactly");
            Assert.AreEqual(1, Store(binding).TotalCount,
                "restoring the original record restores the count");
        }

        [Test]
        public void AStaleOwnerModelOrGateIsRefusedAndNeverCounted()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_stale");
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                Store(binding).TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                                       DateTime.UtcNow).Status);

            var otherModel = new FullHandEnrollmentBinding(Participant, "model_f28_other",
                GateConfigId, new[] { Greeting, ThankYou }, GateAuthorityId);
            Assert.AreEqual(0,
                FullHandParticipantCandidateStore.Open(root, Participant, otherModel).TotalCount,
                "records collected under another model never count");

            var otherGate = new FullHandEnrollmentBinding(Participant, ModelId,
                "config_other_f28", new[] { Greeting, ThankYou }, "gate_other_f28");
            Assert.AreEqual(0,
                FullHandParticipantCandidateStore.Open(root, Participant, otherGate).TotalCount,
                "records collected under another gate never count");

            var foreignBinding = new FullHandEnrollmentBinding(OtherParticipant, ModelId,
                GateConfigId, new[] { Greeting, ThankYou }, GateAuthorityId);
            Assert.Throws<FullHandFormatException>(() =>
                FullHandParticipantCandidateStore.Open(root, Participant, foreignBinding));
            Assert.Throws<FullHandFormatException>(() =>
                FullHandParticipantCandidateStore.Open(root, OtherParticipant, binding));
        }

        [Test]
        public void AnInterruptedWriteAnUnsafeRootAndAFileInThePathNeverCount()
        {
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            FullHandParticipantCandidateStore store = Store(binding);
            // a staging name is what an interrupted write leaves behind
            File.WriteAllText(Path.Combine(RecordDirectory(), "fhc_partial.json.tmp"),
                              "{}", Encoding.UTF8);
            File.WriteAllText(Path.Combine(RecordDirectory(), "not_a_record.json"),
                              "{}", Encoding.UTF8);
            store.Reload();
            Assert.AreEqual(0, store.TotalCount);
            Assert.IsTrue(store.InventoryComplete);

            Assert.Throws<FullHandFormatException>(() =>
                FullHandParticipantCandidateStore.Open("relative\\root", Participant, binding));
            Assert.Throws<FullHandFormatException>(() =>
                FullHandParticipantCandidateStore.Open(
                    Path.Combine(root, "..", "escape"), Participant, binding));

            // a FILE where the learner directory belongs is refused before anything is created
            string otherRoot = Path.Combine(root, "second");
            Directory.CreateDirectory(otherRoot);
            File.WriteAllText(Path.Combine(otherRoot, Participant), "x", Encoding.UTF8);
            Assert.Throws<FullHandFormatException>(() =>
                FullHandParticipantCandidateStore.Open(otherRoot, Participant, binding));
        }

        [Test]
        public void AnIncompleteDirectoryScanEstablishesNoInventoryAndLocksTheGate()
        {
            string wordA, wordB;
            FullHandEnrollmentBinding binding;
            FullHandParticipantCandidateStore store =
                CollectFullRound(out wordA, out wordB, out binding, true);
            Assert.IsTrue(store.AllWordsCollected);
            store.Reload();
            Assert.AreEqual(6, store.TotalCount);
            Assert.AreEqual(0, store.RefusedRecordCount,
                "a healthy record pair reloads with no refusal; its binary sidecar is not " +
                "a rejected record");

            // more files than the accepted inspection bound: the listing cannot be completed
            for (int index = 0; index <= FullHandParticipantCandidateStore.MaxFilesInspected;
                 index++)
                File.WriteAllText(Path.Combine(RecordDirectory(),
                    "pad_" + index.ToString("D5") + ".dat"), "x", Encoding.UTF8);

            store.Reload();
            Assert.IsFalse(store.InventoryComplete);
            Assert.AreEqual(0, store.TotalCount, "an incomplete scan establishes no counts");
            Assert.IsFalse(store.AllWordsCollected,
                "a partial listing must never satisfy the quorum");

            string failure;
            Assert.IsFalse(FullHandEnrollmentQuestGate.Verify(store, out failure));
            StringAssert.Contains(FullHandEnrollmentCodes.InventoryIncomplete, failure);

            Accepted accepted = AcceptedCapture(Direct(), 30, "capture_f28_incomplete");
            Assert.AreEqual(wordA, accepted.Word);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.StorageFailed,
                store.TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                              DateTime.UtcNow).Status,
                "saving into an unestablished inventory is refused");
            Assert.IsFalse(store.InventoryComplete,
                "the rescan attempted by that save found the folder still over budget");

            // the excess is removed: the next save's single bounded rescan recovers, and the
            // recording is genuinely stored. Reload() is deliberately NOT called by the test.
            foreach (string pad in Directory.GetFiles(RecordDirectory(), "pad_*.dat"))
                File.Delete(pad);
            FullHandEnrollmentSaveResult recovered = store.TrySave(
                accepted.Decision, accepted.Evidence, accepted.Word, DateTime.UtcNow);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved, recovered.Status,
                recovered.FailureCode + " " + recovered.FailureDetail);
            Assert.IsTrue(store.InventoryComplete);
            Assert.AreEqual(4, store.CountFor(wordA));
            Assert.AreEqual(7, store.TotalCount);
            Assert.AreEqual(0, store.RefusedRecordCount);
            Assert.IsTrue(FullHandEnrollmentQuestGate.Verify(store, out failure), failure);
        }

        [Test]
        public void AFreshParticipantStartsEmptyAndThePreviousRecordsAreLeftUntouched()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_reset");
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                Store(binding).TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                                       DateTime.UtcNow).Status);
            int filesBefore = Directory.GetFiles(RecordDirectory()).Length;

            const string freshLearner = "participant_f28_fresh";
            var freshBinding = new FullHandEnrollmentBinding(freshLearner, ModelId, GateConfigId,
                new[] { Greeting, ThankYou }, GateAuthorityId);
            FullHandParticipantCandidateStore fresh =
                FullHandParticipantCandidateStore.Open(root, freshLearner, freshBinding);
            Assert.AreEqual(0, fresh.TotalCount, "a fresh participant starts empty");
            Assert.IsFalse(fresh.AllWordsCollected);

            Assert.AreEqual(filesBefore, Directory.GetFiles(RecordDirectory()).Length,
                "the previous participant's files are preserved, never moved or deleted");
            Assert.AreEqual(1, Store(binding).TotalCount);
        }

        [Test]
        public void RecordFileNamesAreDerivedAndCarryNoCaptureOrWordText()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_naming");
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                Store(binding).TrySave(accepted.Decision, accepted.Evidence, accepted.Word,
                                       DateTime.UtcNow).Status);

            foreach (string file in Directory.GetFiles(RecordDirectory()))
            {
                string name = Path.GetFileName(file);
                StringAssert.StartsWith(FullHandParticipantCandidateStore.RecordIdPrefix, name);
                StringAssert.DoesNotContain(accepted.CaptureId, name);
                StringAssert.DoesNotContain(accepted.Word, name);
            }
            Assert.AreEqual(
                FullHandParticipantCandidateStore.BuildRecordId(
                    binding, accepted.Word, accepted.CaptureId,
                    accepted.Evidence.SourceSha256, accepted.Decision.QueryFeatureDigest) +
                    FullHandParticipantCandidateStore.RecordExtension,
                Path.GetFileName(Directory.GetFiles(RecordDirectory(), "*.json")[0]));
        }

        // ---- route selection and configuration ---------------------------------

        [Test]
        public void TheGameplayFlagOwnsTheRouteAndAMissingPipelineRefusesInsteadOfFallingBack()
        {
            string previousGameplay = Environment.GetEnvironmentVariable(
                FullHandGameplayConfiguration.EnableVariable);
            string previousPipeline = Environment.GetEnvironmentVariable(
                FullHandGameCaptureConfiguration.EnableVariable);
            try
            {
                Environment.SetEnvironmentVariable(
                    FullHandGameplayConfiguration.EnableVariable, "1");
                Environment.SetEnvironmentVariable(
                    FullHandGameCaptureConfiguration.EnableVariable, null);

                Assert.IsTrue(FullHandEnrollmentRuntime.ModeActive,
                    "the gameplay flag alone selects the full-hand route");
                FullHandEnrollmentBinding binding;
                string code, detail;
                Assert.IsFalse(FullHandEnrollmentBindingFactory.TryCreateForCurrentParticipant(
                    out binding, out code, out detail));
                Assert.AreEqual(FullHandEnrollmentCodes.PipelineNotEnabled, code,
                    "a missing pipeline refuses the full-hand route rather than selecting legacy");
                Assert.IsNull(binding);

                string failure;
                Assert.IsFalse(FullHandEnrollmentQuestGate.VerifyForIdentity(
                    new PetRobotLearnerRootIdentity(Participant), out failure));
                StringAssert.Contains(FullHandEnrollmentCodes.PipelineNotEnabled, failure);
            }
            finally
            {
                Environment.SetEnvironmentVariable(
                    FullHandGameplayConfiguration.EnableVariable, previousGameplay);
                Environment.SetEnvironmentVariable(
                    FullHandGameCaptureConfiguration.EnableVariable, previousPipeline);
            }
        }

        [Test]
        public void AnUnconfiguredProcessFailsClosedInsteadOfClaimingReadiness()
        {
            // the opt-ins are off in the EditMode process, so nothing may be pinned
            FullHandEnrollmentBinding binding;
            string code, detail;
            Assert.IsFalse(FullHandEnrollmentBindingFactory.TryCreateForCurrentParticipant(
                out binding, out code, out detail));
            Assert.AreEqual(FullHandEnrollmentCodes.ModeDisabled, code);
            Assert.IsNull(binding);
            Assert.IsFalse(FullHandEnrollmentRuntime.ModeActive);

            string failure;
            Assert.IsFalse(FullHandEnrollmentQuestGate.Verify(null, out failure));
            Assert.AreEqual(FullHandEnrollmentCodes.StoreUnavailable, failure);
        }

        // ---- the REAL production world gate -------------------------------------

        [Test]
        public void TheProductionQuestGateVerifierReadsTheRealStoreCountsAndNotTheLegacyOnes()
        {
            string wordA, wordB;
            FullHandEnrollmentBinding binding;
            FullHandParticipantCandidateStore store =
                CollectFullRound(out wordA, out wordB, out binding, false);

            // the identity carries a root that no legacy repository could ever satisfy, so a
            // legacy inventory answer would be observable as a failure here
            var identity = new PetRobotLearnerRootIdentity(
                Participant, Path.Combine(root, "no_such_legacy_root"));
            FullHandEnrollmentRuntime.OverrideForTests(true, binding, store);

            var verifier = new ProductionQuestGateVerifier(identity);
            Assert.IsFalse(verifier.VerifyOverallEnrollmentGateReady(),
                "a partially collected store must never open the quest");
            StringAssert.Contains(wordB, verifier.LastFailure);

            CollectFor(store, wordB, 3);
            Assert.IsTrue(verifier.VerifyOverallEnrollmentGateReady(), verifier.LastFailure);
            Assert.AreEqual(string.Empty, verifier.LastFailure);

            // a stale identity is refused before any store is read
            var stale = new PetRobotLearnerRootIdentity(OtherParticipant);
            var staleVerifier = new ProductionQuestGateVerifier(stale);
            Assert.IsFalse(staleVerifier.VerifyOverallEnrollmentGateReady());
            StringAssert.Contains(FullHandEnrollmentCodes.ParticipantChanged,
                                  staleVerifier.LastFailure);
        }

        // ---- the real session ---------------------------------------------------

        sealed class FakeAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            public PracticeRecognitionAttempt Next;
            public string Expected = Greeting;

            public PersonalLiveCaptureState CaptureState
            { get { return PersonalLiveCaptureState.Ready; } }
            public string CaptureSessionId
            { get { return Next == null ? string.Empty : Next.CaptureSessionId; } }
            public string ActiveProfileId { get { return "f28_fake_profile"; } }
            public bool CandidateRouteActive { get { return true; } }
            public bool SetExpectedWord(string word) { Expected = word; return true; }
            public bool BeginCapture() { return true; }
            public PracticeRecognitionAttempt FinalizeCapture() { return Next; }
            public bool ResetCapture() { return true; }
        }

        sealed class RecordingRepository : IPersonalizationCandidateRepository
        {
            public int SaveCalls;
            public string RootDirectory { get { return string.Empty; } }
            public string LearnerId { get { return Participant; } }
            public CandidateCountSnapshot Counts { get { return CandidateCountSnapshot.Empty; } }
            public CandidateCountSnapshot ReloadAndVerify() { return CandidateCountSnapshot.Empty; }

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                throw new InvalidOperationException(
                    "the full-hand route must never reach the 16-point candidate store");
            }
        }

        static PracticeRecognitionAttempt AttemptFor(Accepted accepted, string captureId)
        {
            return FullHandGameplayAttemptFactory.BuildAttempt(
                accepted.Decision, captureId, PersonalLiveCaptureState.Failed, null,
                "f28_fake_profile", null, null, accepted.Evidence);
        }

        static PracticeRecognitionAttempt TechnicalAttempt(string word, string captureId)
        {
            FullHandGameplayDecision refusal =
                FullHandGameplayEvaluator.Decide(null, word, captureId, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, refusal.Kind);
            return FullHandGameplayAttemptFactory.BuildAttempt(
                refusal, captureId, PersonalLiveCaptureState.Failed, null,
                "f28_fake_profile", null, null, null);
        }

        static void Run(PetRobotSignPracticeSession session, FakeAdapter adapter,
                        PracticeRecognitionAttempt attempt)
        {
            adapter.Next = attempt;
            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());
        }

        string WordForOrder(bool swapped)
        {
            return ProbeTop1(swapped ? Swapped() : Direct(), 6);
        }

        /// <summary>
        /// Fills one word from PHYSICALLY DISJOINT synthetic captures. The label ordering only
        /// changes which class name the same bank ranks first; the stored record is keyed by
        /// SOURCE and FEATURE identity, so the two words must be filled from different raw
        /// captures or the second one is a correct duplicate refusal. Frame counts 12..14 and
        /// 24..26 are used here; 30 is reserved for the recovery capture.
        /// </summary>
        void CollectFor(FullHandParticipantCandidateStore store, string word, int count)
        {
            bool direct = string.Equals(word, WordForOrder(false), StringComparison.Ordinal);
            FullHandGameCaptureProcessor processor = direct ? Direct() : Swapped();
            int frameBase = direct ? 12 : 24;
            for (int index = 0; index < count; index++)
            {
                Accepted accepted = AcceptedCapture(processor, frameBase + index,
                    "capture_f28_fill_" + word + "_" + index);
                Assert.AreEqual(word, accepted.Word);
                // explicit: every fill capture is a distinct source and a distinct feature
                Assert.IsTrue(filledSourceHashes.Add(accepted.Evidence.SourceSha256),
                    "the fixture must fill from a source it has not used before");
                Assert.IsTrue(filledFeatureDigests.Add(accepted.Decision.QueryFeatureDigest),
                    "the fixture must fill from a feature digest it has not used before");
                FullHandEnrollmentSaveResult save = store.TrySave(
                    accepted.Decision, accepted.Evidence, accepted.Word, DateTime.UtcNow);
                Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved, save.Status,
                    save.FailureCode + " " + save.FailureDetail);
            }
        }

        FullHandParticipantCandidateStore CollectFullRound(out string wordA, out string wordB,
                                                           out FullHandEnrollmentBinding binding,
                                                           bool includeSecondWord)
        {
            wordA = WordForOrder(false);
            wordB = WordForOrder(true);
            Assert.AreNotEqual(wordA, wordB,
                "the fixture needs one accepted word from each label ordering");
            binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);
            CollectFor(store, wordA, 3);
            if (includeSecondWord) CollectFor(store, wordB, 3);
            return store;
        }

        [Test]
        public void TheSessionCollectsThreeRecordsPerWordBeforeItAdvances()
        {
            string wordA = WordForOrder(false);
            string wordB = WordForOrder(true);
            Assert.AreNotEqual(wordA, wordB);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);

            var adapter = new FakeAdapter();
            var repository = new RecordingRepository();
            FullHandGameCaptureProcessor processor = Direct();
            var accepted = new Accepted[3];
            for (int index = 0; index < accepted.Length; index++)
            {
                accepted[index] = AcceptedCapture(processor, 6 + index,
                                                  "capture_f28_wordA_" + index);
                Assert.AreEqual(wordA, accepted[index].Word,
                    "this fixture expects the synthetic bank to rank one stable class here");
            }

            using (var session = new PetRobotSignPracticeSession(
                adapter, repository, binding, store))
            {
                Assert.IsTrue(session.FullHandRecordCollectionActive);
                Assert.AreEqual(2, session.Words.Count);
                Assert.AreEqual(wordA, session.CurrentWord);
                Assert.AreEqual(3, session.FullHandRequiredRecordCount);

                Run(session, adapter, AttemptFor(accepted[0], accepted[0].CaptureId));
                Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                                session.LastFullHandSaveResult.Status);
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                Assert.IsFalse(session.ContinueToNextWord(),
                    "the round must not advance before the records are collected");
                Assert.IsFalse(session.CanContinueWithNeedsPractice,
                    "a record round has no skip-to-next-word path");
                Assert.IsTrue(session.CanRetry);
                Assert.IsTrue(session.Retry());
                Assert.AreEqual(wordA, session.CurrentWord);

                Run(session, adapter, AttemptFor(accepted[1], accepted[1].CaptureId));
                Assert.AreEqual(2, session.CurrentFullHandRecordCount);
                Assert.IsTrue(session.Retry());

                Run(session, adapter, AttemptFor(accepted[2], accepted[2].CaptureId));
                Assert.AreEqual(3, session.CurrentFullHandRecordCount);
                Assert.IsTrue(session.IsCurrentFullHandWordCollected);
                Assert.AreEqual(PetRobotWordLearningStatus.Mastered,
                                session.GetWordLearningStatus(wordA));
                Assert.AreEqual(3, session.FullHandRecordTotal);
                Assert.IsTrue(session.ContinueToNextWord());
                Assert.AreEqual(wordB, session.CurrentWord);
                Assert.AreEqual(0, session.CurrentFullHandRecordCount);

                Assert.AreEqual(0, repository.SaveCalls,
                    "no 16-point personalization candidate is written on this route");
                Assert.IsFalse(session.TrustedEnrollmentEnabled);
                Assert.IsFalse(session.OverallEnrollmentGateReady);
            }
            Assert.AreEqual(3, store.CountFor(wordA));
        }

        [Test]
        public void ADuplicateAndAFailedSaveRenderHonestCopyAndMakeNoProgress()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_replay");

            var adapter = new FakeAdapter();
            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                Assert.IsTrue(session.Retry());

                // a NEW capture identity carrying the SAME evidence adds nothing
                Run(session, adapter, AttemptFor(accepted, "capture_f28_replay_again"));
                Assert.AreEqual(FullHandEnrollmentSaveStatus.Duplicate,
                                session.LastFullHandSaveResult.Status);
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                MoaStoryPresentation duplicate = SessionStory(session,
                    MoaStoryFeedbackKind.StrictSuccess);
                StringAssert.Contains("저장하지 않았어", duplicate.BodyText);
                StringAssert.Contains("연습 기록 1/3", duplicate.BodyText);
                Assert.AreEqual(MoaStoryAction.RetryEnrollment, duplicate.Action);
                Assert.IsTrue(session.Retry());

                // the store's own directory disappears: the write cannot succeed
                Directory.Delete(RecordDirectory(), true);
                Accepted second = AcceptedCapture(Direct(), 9, "capture_f28_iofail");
                Assert.AreEqual(wordA, second.Word);
                Run(session, adapter, AttemptFor(second, second.CaptureId));
                Assert.AreNotEqual(FullHandEnrollmentSaveStatus.Saved,
                                   session.LastFullHandSaveResult.Status);
                Assert.AreEqual(1, session.CurrentFullHandRecordCount,
                    "a failed write never moves the count");
                MoaStoryPresentation failed = SessionStory(session,
                    MoaStoryFeedbackKind.StrictSuccess);
                StringAssert.Contains("저장하지는 못했어", failed.BodyText,
                    "a failed save must say so instead of reporting progress");
                Assert.AreEqual(MoaStoryAction.RetryEnrollment, failed.Action);
                Assert.IsTrue(session.CanRetry);
            }
        }

        [Test]
        public void ABoundRecordSessionRefusesALegacyShapedAttemptWithoutAnyLegacyWrite()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();
            var repository = new RecordingRepository();

            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed, true, wordA, wordA,
                true, true, true, true, true);
            var legacy = new PracticeRecognitionAttempt(
                "capture_f28_legacy_shaped", wordA, wordA, PersonalLiveCaptureState.Completed,
                outcome, null, null, PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "f28_fake_profile", 0.5d, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
            Assert.IsNull(legacy.FullHandGameplay);

            using (var session = new PetRobotSignPracticeSession(
                adapter, repository, binding, store))
            {
                Run(session, adapter, legacy);
                Assert.AreEqual(0, repository.SaveCalls,
                    "a bound record session never reaches the 16-point candidate store");
                Assert.AreEqual(0, session.CurrentFullHandRecordCount);
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA));
                Assert.AreEqual(FullHandEnrollmentCopy.UnexpectedResult,
                                session.LastFullHandCollectionMessage);
                Assert.AreEqual(FullHandEnrollmentSaveStatus.NotRequested,
                                session.LastFullHandSaveResult.Status);
                Assert.AreEqual(0, store.TotalCount);
            }
        }

        [Test]
        public void AReopenedSessionContinuesFromThePersistedCounts()
        {
            string wordA, wordB;
            FullHandEnrollmentBinding binding;
            CollectFullRound(out wordA, out wordB, out binding, false);

            using (var session = new PetRobotSignPracticeSession(
                new FakeAdapter(), new RecordingRepository(), binding, Store(binding)))
            {
                Assert.AreEqual(wordB, session.CurrentWord,
                    "a fully collected word is skipped on reopen");
                Assert.AreEqual(3, session.GetFullHandRecordCount(wordA));
                Assert.AreEqual(0, session.GetFullHandRecordCount(wordB));
                Assert.AreEqual(PetRobotWordLearningStatus.Mastered,
                                session.GetWordLearningStatus(wordA));
            }
        }

        // ---- F29: the practice attempt budget -----------------------------------

        /// <summary>F29: one finalized WRONG-WORD full-hand attempt. Nothing is stored.</summary>
        PracticeRecognitionAttempt WrongWordAttempt(string captureId, int frames)
        {
            FullHandGameCaptureProcessor processor = Direct();
            string other = OtherWord(ProbeTop1(processor, frames));
            FullHandGameCaptureResult evidence =
                processor.Process(captureId, other, RawCapture(frames));
            FullHandGameplayDecision decision =
                FullHandGameplayEvaluator.Decide(evidence, other, captureId, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.AcceptedDifferentWord, decision.Kind);
            return FullHandGameplayAttemptFactory.BuildAttempt(
                decision, captureId, PersonalLiveCaptureState.Failed, null,
                "f28_fake_profile", null, null, evidence);
        }

        /// <summary>F29: one finalized valid ABSTENTION. Not a wrong word, not a fault.</summary>
        PracticeRecognitionAttempt AbstainAttempt(string word, string captureId, int frames)
        {
            FullHandGameCaptureProcessor abstaining =
                Processor(new[] { Greeting, ThankYou }, RejectAllGates());
            FullHandGameCaptureResult evidence =
                abstaining.Process(captureId, word, RawCapture(frames));
            FullHandGameplayDecision decision =
                FullHandGameplayEvaluator.Decide(evidence, word, captureId, true);
            Assert.AreEqual(FullHandGameplayDecisionKind.NotDistinguished, decision.Kind);
            return FullHandGameplayAttemptFactory.BuildAttempt(
                decision, captureId, PersonalLiveCaptureState.Failed, null,
                "f28_fake_profile", null, null, evidence);
        }

        [Test]
        public void F29TenWrongWordOrAbstainedAttemptsDeferTheWordWithNoRecord()
        {
            // the wrong-word attempts below declare OtherWord(Direct top1) as their target,
            // so THIS round practises exactly that word
            string wordA = OtherWord(WordForOrder(false));
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                for (int index = 0; index < 10; index++)
                {
                    Assert.AreEqual(index, session.CurrentFullHandAttemptCount);
                    Run(session, adapter, index % 2 == 0
                        ? WrongWordAttempt("capture_f29_wrong_" + index, 6)
                        : AbstainAttempt(wordA, "capture_f29_abstain_" + index, 6));
                    if (index < 9)
                        Assert.IsTrue(session.Retry(), "the budget is not spent yet");
                }

                Assert.AreEqual(10, session.CurrentFullHandAttemptCount);
                Assert.IsTrue(session.IsCurrentFullHandAttemptCapReached);
                Assert.AreEqual(0, session.CurrentFullHandRecordCount);
                Assert.AreEqual(0, store.TotalCount);
                Assert.IsFalse(session.CanRetry);
                Assert.IsFalse(session.Retry(), "Retry cannot bypass the attempt budget");
                Assert.IsFalse(session.StartExpression(),
                    "a new manual capture cannot bypass the attempt budget");
                Assert.IsFalse(session.PrepareAutoAttempt(),
                    "a new auto attempt cannot bypass the attempt budget");
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA));

                // the cap copy wins over BOTH early full-hand paths
                foreach (MoaStoryFeedbackKind kind in new[]
                         { MoaStoryFeedbackKind.NotDistinguishedRetry,
                           MoaStoryFeedbackKind.WrongWord })
                {
                    MoaStoryPresentation capped = SessionStory(session, kind);
                    Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, capped.Action);
                    Assert.IsTrue(capped.ActionEnabled);
                    // F32: another word of this visit is still waiting, so the button says so
                    Assert.AreEqual("다음 표현으로", capped.ActionLabel);
                    StringAssert.Contains("10번까지", capped.BodyText);
                    StringAssert.Contains("연습 기록 0/3", capped.BodyText);
                    StringAssert.DoesNotContain(MoaStoryGuidedPracticeFlow.NotDistinguishedCopy,
                                                capped.BodyText);
                    StringAssert.DoesNotContain(FullHandEnrollmentCopy.TechnicalStop,
                                                capped.BodyText);
                    StringAssert.DoesNotContain(FullHandEnrollmentCopy.Collected,
                                                capped.BodyText);
                }

                Assert.IsTrue(session.CanLeaveRoundIncomplete);
                // F32: the exhausted word is deferred for THIS visit; the visit continues
                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
                Assert.AreEqual(OtherWord(wordA), session.CurrentWord);
                Assert.IsTrue(session.IsFullHandWordDeferred(wordA));
                Assert.AreEqual(0, session.CurrentFullHandAttemptCount,
                    "the next word starts with its own budget");
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA),
                    "a deferred word is never marked mastered");
                Assert.AreEqual(0, session.GetFullHandRecordCount(wordA));
                Assert.IsFalse(session.AllFullHandRecordsCollected);
                Assert.IsFalse(session.OverallEnrollmentGateReady,
                    "the NPC gate stays locked until the REAL records are collected");
            }
            string failure;
            Assert.IsFalse(FullHandEnrollmentQuestGate.Verify(store, out failure));
        }

        [Test]
        public void F29AnAcceptedButUnsavedAttemptCannotLoopPastTheCapOrCountAsARecord()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f29_first");
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                Assert.AreEqual(1, session.CurrentFullHandAttemptCount);

                // the SAME evidence under nine NEW capture identities: accepted every time,
                // stored never, and each one still spends one attempt
                for (int index = 0; index < 9; index++)
                {
                    Assert.IsTrue(session.Retry());
                    Run(session, adapter,
                        AttemptFor(accepted, "capture_f29_replay_" + index));
                    Assert.AreEqual(FullHandEnrollmentSaveStatus.Duplicate,
                                    session.LastFullHandSaveResult.Status);
                    Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                }

                Assert.AreEqual(10, session.CurrentFullHandAttemptCount);
                Assert.IsTrue(session.IsCurrentFullHandAttemptCapReached);
                Assert.IsFalse(session.Retry());
                Assert.IsFalse(session.StartExpression());
                Assert.AreEqual(1, store.CountFor(wordA),
                    "a duplicate never becomes a second or third record");
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA));

                MoaStoryPresentation capped = SessionStory(session,
                    MoaStoryFeedbackKind.StrictSuccess);
                Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, capped.Action,
                    "the cap wins over the success-but-not-saved retry path");
                StringAssert.Contains("연습 기록 1/3", capped.BodyText);
            }
        }

        [Test]
        public void F29AThirdDistinctRecordOnTheTenthAttemptStillSucceeds()
        {
            string wordA = WordForOrder(false);
            string wordB = WordForOrder(true);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);
            FullHandGameCaptureProcessor processor = Direct();
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                for (int index = 0; index < 7; index++)
                {
                    Run(session, adapter,
                        AbstainAttempt(wordA, "capture_f29_pre_" + index, 6));
                    Assert.IsTrue(session.Retry());
                }
                Assert.AreEqual(7, session.CurrentFullHandAttemptCount);

                for (int index = 0; index < 3; index++)
                {
                    Accepted accepted = AcceptedCapture(processor, 6 + index,
                                                        "capture_f29_late_" + index);
                    Assert.AreEqual(wordA, accepted.Word);
                    Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                    Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                                    session.LastFullHandSaveResult.Status);
                    if (index < 2) Assert.IsTrue(session.Retry());
                }

                Assert.AreEqual(10, session.CurrentFullHandAttemptCount);
                Assert.AreEqual(3, session.CurrentFullHandRecordCount);
                Assert.IsFalse(session.IsCurrentFullHandAttemptCapReached,
                    "the tenth attempt supplied the last record, so the word is collected");
                Assert.IsTrue(session.IsCurrentFullHandWordCollected);

                MoaStoryPresentation success = SessionStory(session,
                    MoaStoryFeedbackKind.StrictSuccess);
                Assert.AreEqual(MoaStoryAction.ContinueToNextWord, success.Action);
                StringAssert.Contains(FullHandEnrollmentCopy.Collected, success.BodyText);

                Assert.IsTrue(session.ContinueToNextWord());
                Assert.AreEqual(wordB, session.CurrentWord);
                Assert.AreEqual(0, session.CurrentFullHandAttemptCount,
                    "each word of the round has its own budget");
            }
        }

        [Test]
        public void F29AnEarlyThirdRecordStopsTheWordBeforeTheBudgetIsSpent()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            FullHandGameCaptureProcessor processor = Direct();
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                for (int index = 0; index < 3; index++)
                {
                    Accepted accepted = AcceptedCapture(processor, 6 + index,
                                                        "capture_f29_early_" + index);
                    Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                    if (index < 2) Assert.IsTrue(session.Retry());
                }
                Assert.AreEqual(3, session.CurrentFullHandAttemptCount);
                Assert.AreEqual(3, session.CurrentFullHandRecordCount);
                Assert.IsTrue(session.IsCurrentFullHandWordCollected);
                Assert.IsFalse(session.IsCurrentFullHandAttemptCapReached);
                Assert.IsFalse(session.Retry(),
                    "a collected word is not practised again in this round");
                Assert.IsFalse(session.CanLeaveRoundIncomplete);
            }
        }

        [Test]
        public void F29ANewSessionResumesTheRecordsWithAFreshAttemptBudget()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore first = Store(binding);
            CollectFor(first, wordA, 1);
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, first))
            {
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                for (int index = 0; index < 10; index++)
                {
                    Run(session, adapter,
                        AbstainAttempt(wordA, "capture_f29_spend_" + index, 6));
                    if (index < 9) Assert.IsTrue(session.Retry());
                }
                Assert.IsTrue(session.IsCurrentFullHandAttemptCapReached);
                Assert.IsTrue(session.LeaveRoundIncomplete());
            }

            using (var resumed = new PetRobotSignPracticeSession(
                new FakeAdapter(), new RecordingRepository(), binding, Store(binding)))
            {
                Assert.AreEqual(wordA, resumed.CurrentWord);
                Assert.AreEqual(1, resumed.CurrentFullHandRecordCount,
                    "the records already collected are kept");
                Assert.AreEqual(0, resumed.CurrentFullHandAttemptCount,
                    "a later visit starts a fresh attempt budget");
                Assert.IsFalse(resumed.IsCurrentFullHandAttemptCapReached);
                Assert.IsTrue(resumed.PrepareAutoAttempt());
            }
        }

        [Test]
        public void F29AMalformedBlankCaptureIdSpendsAnAttemptAndReachesTheTechnicalStop()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();

            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed, true, wordA, wordA,
                true, true, true, true, true);
            // a finalized result carrying NO capture identity: malformed, not a duplicate
            var malformed = new PracticeRecognitionAttempt(
                string.Empty, wordA, wordA, PersonalLiveCaptureState.Completed,
                outcome, null, null, PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "f28_fake_profile", 0.5d, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                for (int index = 0; index < 3; index++)
                {
                    Run(session, adapter, malformed);
                    Assert.AreEqual(index + 1, session.CurrentFullHandAttemptCount,
                        "a malformed finalization spends one attempt, it is not free");
                    if (index < 2) Assert.IsTrue(session.Retry());
                }

                Assert.AreEqual(3, session.GetConsecutiveTechnicalFailureCount(wordA));
                Assert.IsFalse(session.CanRetry);
                Assert.IsFalse(session.Retry(),
                    "a malformed result cannot be retried without end");
                Assert.AreEqual(0, session.CurrentFullHandRecordCount);
                Assert.AreEqual(0, store.TotalCount);
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA));
                // this malformed attempt carries a StrictSuccess GUIDED label; the UI must not
                // offer a retry the session refuses
                foreach (MoaStoryFeedbackKind label in new[]
                         { MoaStoryFeedbackKind.StrictSuccess,
                           MoaStoryFeedbackKind.TechnicalFailure })
                {
                    MoaStoryPresentation stopped = SessionStory(session, label);
                    Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, stopped.Action);
                    Assert.IsTrue(stopped.ActionEnabled);
                    // F32: a second word is still waiting, so the copy and the label move on
                    Assert.AreEqual("다음 표현으로", stopped.ActionLabel);
                    StringAssert.Contains(FullHandEnrollmentCopy.TechnicalStopMovingOn,
                                          stopped.BodyText);
                    StringAssert.DoesNotContain(FullHandEnrollmentCopy.Collected,
                                                stopped.BodyText);
                }

                Assert.IsTrue(session.CanLeaveRoundIncomplete);
                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
                Assert.AreEqual(OtherWord(wordA), session.CurrentWord);
                Assert.IsFalse(session.OverallEnrollmentGateReady);
            }
        }

        [Test]
        public void F29ADuplicateCallbackAndARejectedBeginSpendNoAttempt()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f29_guard");
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.AreEqual(1, session.CurrentFullHandAttemptCount);
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);

                Assert.IsFalse(session.StartExpression(),
                    "a refused BeginCapture is not a finalized attempt");
                Assert.AreEqual(1, session.CurrentFullHandAttemptCount,
                    "a rejected begin spends no attempt");

                // the SAME capture identity arriving a second time
                Assert.IsTrue(session.Retry());
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.AreEqual(1, session.CurrentFullHandAttemptCount,
                    "a duplicate callback spends no extra attempt");
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                Assert.AreEqual(1, store.CountFor(wordA));
            }
        }

        // ---- the world snapshot -------------------------------------------------

        [Test]
        public void ACompletedTwoWordRoundReportsItsActualWordsAndRecordTotal()
        {
            string wordA, wordB;
            FullHandEnrollmentBinding binding;
            CollectFullRound(out wordA, out wordB, out binding, true);

            using (var session = new PetRobotSignPracticeSession(
                new FakeAdapter(), new RecordingRepository(), binding, Store(binding)))
            {
                Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
                Assert.IsTrue(session.AllFullHandRecordsCollected);
                Assert.IsTrue(session.OverallEnrollmentGateReady);
                Assert.AreEqual(6, session.FullHandRecordTotal);

                PetRobotPracticeCompletionSnapshot snapshot;
                Assert.IsTrue(PetRobotPracticeCompletionSnapshot.TryCreate(session, out snapshot));
                Assert.IsTrue(snapshot.IsFullHandRecordRound);
                Assert.AreEqual(2, snapshot.Words.Count);
                Assert.AreEqual(2, snapshot.MasteredCount);
                Assert.AreEqual(0, snapshot.NeedsPracticeCount,
                    "a word outside the current model vocabulary is absent, not failed practice");
                Assert.AreEqual(6, snapshot.FullHandRecordTotal);
                Assert.IsTrue(snapshot.AllRecordsCollected);

                PetRobotWordLearningStatus love;
                Assert.IsFalse(snapshot.TryGetWordStatus("LOVE", out love),
                    "LOVE was not part of this round and is not reported at all");
                Assert.AreEqual(PetRobotWordLearningStatus.Mastered,
                                snapshot.GetWordStatus(wordA));
                StringAssert.Contains("6", snapshot.WorldSummaryMessage);
                StringAssert.DoesNotContain("복습 필요", snapshot.WorldSummaryMessage);
            }
        }

        [Test]
        public void TheLegacyThreeWordSnapshotIsUnchanged()
        {
            var legacy = new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.NeedsPractice,
                PetRobotWordLearningStatus.Mastered, 2, 1, 7);
            Assert.IsFalse(legacy.IsFullHandRecordRound);
            Assert.AreEqual(3, legacy.Words.Count);
            Assert.AreEqual(7, legacy.CandidateSavedTotal);
            Assert.AreEqual(PetRobotWordLearningStatus.NeedsPractice,
                            legacy.GetWordStatus("LOVE"));
            Assert.AreEqual("연습 완료 · 1개 복습 필요", legacy.WorldSummaryMessage);
        }

        [Test]
        public void ATechnicalStopOnTheLastAvailableWordEndsTheVisitHonestly()
        {
            string wordA = WordForOrder(false);
            string wordB = OtherWord(wordA);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                for (int index = 0; index < 3; index++)
                {
                    Run(session, adapter, TechnicalAttempt(wordA, "capture_f28_tech_" + index));
                    if (index < 2) Assert.IsTrue(session.Retry());
                }
                Assert.IsFalse(session.CanRetry,
                    "three consecutive technical failures stop this word");
                Assert.IsFalse(session.Retry(),
                    "a direct Retry cannot bypass the technical stop");
                Assert.AreEqual(0, session.CurrentFullHandRecordCount);
                Assert.AreEqual(0, store.TotalCount);
                Assert.IsFalse(session.CanContinueWithNeedsPractice,
                    "a technical stop must not be able to finish the round as practised");
                Assert.IsTrue(session.CanLeaveRoundIncomplete);

                // F32: the first word is deferred and the visit continues with the second
                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
                Assert.AreEqual(wordB, session.CurrentWord);

                // the LAST available word also stops technically: now the visit ends
                for (int index = 0; index < 3; index++)
                {
                    Run(session, adapter, TechnicalAttempt(wordB, "capture_f32_tech_b_" + index));
                    if (index < 2) Assert.IsTrue(session.Retry());
                }
                Assert.IsTrue(session.CanLeaveRoundIncomplete);

                MoaStoryPresentation stop = SessionStory(session,
                    MoaStoryFeedbackKind.TechnicalFailure);
                Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, stop.Action);
                Assert.AreEqual("이번 라운드 마무리", stop.ActionLabel,
                    "no word is left, so the label may say the round is being wrapped up");
                StringAssert.Contains(FullHandEnrollmentCopy.TechnicalStop, stop.BodyText);
                StringAssert.DoesNotContain(FullHandEnrollmentCopy.Collected, stop.BodyText);

                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA),
                    "nothing is marked mastered by leaving early");
                Assert.IsFalse(session.AllFullHandRecordsCollected);
                Assert.IsFalse(session.OverallEnrollmentGateReady,
                    "the NPC gate stays locked after an unfinished round");

                PetRobotPracticeCompletionSnapshot snapshot;
                Assert.IsTrue(PetRobotPracticeCompletionSnapshot.TryCreate(session, out snapshot));
                Assert.IsFalse(snapshot.AllRecordsCollected);
                Assert.AreEqual(0, snapshot.MasteredCount);
                Assert.AreEqual(0, snapshot.FullHandRecordTotal);
                StringAssert.Contains("미완료", snapshot.WorldSummaryMessage);

                string failure;
                Assert.IsFalse(FullHandEnrollmentQuestGate.Verify(store, out failure));
            }
        }

        // ---- F32: per-visit deferral of an exhausted word -----------------------

        [Test]
        public void F32ADeferredWordLetsTheOtherWordFinishAndNeverReturnsInThisVisit()
        {
            string wordA = WordForOrder(false);
            string wordB = WordForOrder(true);
            Assert.AreNotEqual(wordA, wordB);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                // one REAL record for wordA, then the rest of the budget is spent
                Accepted first = AcceptedCapture(Direct(), 6, "capture_f32_a0");
                Assert.AreEqual(wordA, first.Word);
                Run(session, adapter, AttemptFor(first, first.CaptureId));
                Assert.AreEqual(1, session.CurrentFullHandRecordCount);
                for (int index = 1; index < 10; index++)
                {
                    Assert.IsTrue(session.Retry());
                    Run(session, adapter, AbstainAttempt(wordA, "capture_f32_a_" + index, 6));
                }
                Assert.IsTrue(session.IsCurrentFullHandAttemptCapReached);

                MoaStoryPresentation capped = SessionStory(session,
                    MoaStoryFeedbackKind.NotDistinguishedRetry);
                Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, capped.Action);
                Assert.AreEqual("다음 표현으로", capped.ActionLabel);
                StringAssert.Contains("연습 기록 1/3", capped.BodyText);
                StringAssert.DoesNotContain(FullHandEnrollmentCopy.Collected, capped.BodyText);

                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
                Assert.AreEqual(wordB, session.CurrentWord);
                Assert.AreEqual(0, session.CurrentFullHandAttemptCount);
                Assert.AreEqual(1, session.GetFullHandRecordCount(wordA),
                    "the deferred word keeps every record it really collected");

                FullHandGameCaptureProcessor swapped = Swapped();
                for (int index = 0; index < 3; index++)
                {
                    Accepted accepted = AcceptedCapture(swapped, 24 + index,
                                                        "capture_f32_b_" + index);
                    Assert.AreEqual(wordB, accepted.Word);
                    Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                    Assert.AreEqual(FullHandEnrollmentSaveStatus.Saved,
                                    session.LastFullHandSaveResult.Status);
                    if (index < 2) Assert.IsTrue(session.Retry());
                }
                Assert.IsTrue(session.IsCurrentFullHandWordCollected);

                // the deferred word does NOT come back after the other word succeeded
                Assert.IsTrue(session.ContinueToNextWord());
                Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(wordA));
                Assert.IsFalse(session.AllFullHandRecordsCollected);
                Assert.IsFalse(session.OverallEnrollmentGateReady,
                    "one collected word never opens the NPC gate");

                PetRobotPracticeCompletionSnapshot snapshot;
                Assert.IsTrue(PetRobotPracticeCompletionSnapshot.TryCreate(session, out snapshot));
                Assert.IsFalse(snapshot.AllRecordsCollected);
                Assert.AreEqual(1, snapshot.MasteredCount);
                Assert.AreEqual(4, snapshot.FullHandRecordTotal);
            }
            string failure;
            Assert.IsFalse(FullHandEnrollmentQuestGate.Verify(store, out failure));
        }

        [Test]
        public void F32WhenEveryWordIsExhaustedTheVisitEndsIncompleteAndTheGateStaysLocked()
        {
            string wordA = WordForOrder(false);
            string wordB = OtherWord(wordA);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                for (int index = 0; index < 10; index++)
                {
                    Run(session, adapter, AbstainAttempt(wordA, "capture_f32_x_a_" + index, 6));
                    if (index < 9) Assert.IsTrue(session.Retry());
                }
                Assert.IsTrue(session.IsCurrentFullHandAttemptCapReached);

                // the FIRST capped word: another word is really left, so the view and the
                // label promise the next expression
                Assert.IsTrue(session.HasNextEligibleFullHandWord);
                Assert.IsTrue(session.FullHandRecordView.MovesToNextWord);
                MoaStoryPresentation firstCapped = SessionStory(session,
                    MoaStoryFeedbackKind.NotDistinguishedRetry);
                Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, firstCapped.Action);
                Assert.AreEqual("다음 표현으로", firstCapped.ActionLabel);

                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(wordB, session.CurrentWord);

                for (int index = 0; index < 10; index++)
                {
                    Run(session, adapter, AbstainAttempt(wordB, "capture_f32_x_b_" + index, 7));
                    if (index < 9) Assert.IsTrue(session.Retry());
                }
                Assert.IsTrue(session.IsCurrentFullHandAttemptCapReached);

                // the LAST capped word: no other word is left, so nothing may promise a
                // next expression before LeaveRoundIncomplete ends the visit
                Assert.IsFalse(session.HasNextEligibleFullHandWord,
                    "the current word itself is not a next word");
                Assert.IsFalse(session.FullHandRecordView.MovesToNextWord);
                MoaStoryPresentation last = SessionStory(session,
                    MoaStoryFeedbackKind.NotDistinguishedRetry);
                Assert.AreEqual(MoaStoryAction.LeaveRoundIncomplete, last.Action);
                Assert.AreEqual("이번 라운드 마무리", last.ActionLabel);

                Assert.IsTrue(session.LeaveRoundIncomplete());
                Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
                Assert.AreEqual(0, store.TotalCount);
                Assert.IsFalse(session.OverallEnrollmentGateReady);

                PetRobotPracticeCompletionSnapshot snapshot;
                Assert.IsTrue(PetRobotPracticeCompletionSnapshot.TryCreate(session, out snapshot));
                Assert.IsFalse(snapshot.AllRecordsCollected);
                Assert.AreEqual(0, snapshot.MasteredCount);
                Assert.AreEqual(0, snapshot.FullHandRecordTotal);
                StringAssert.Contains("미완료", snapshot.WorldSummaryMessage);
            }
            string failure;
            Assert.IsFalse(FullHandEnrollmentQuestGate.Verify(store, out failure));
        }

        [Test]
        public void F32AFreshSessionClearsThePerVisitDeferralAndKeepsTheSavedRecords()
        {
            string wordA = WordForOrder(false);
            string wordB = OtherWord(wordA);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore first = Store(binding);
            CollectFor(first, wordA, 1);

            var adapter = new FakeAdapter();
            using (var visit = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, Store(binding)))
            {
                Assert.AreEqual(1, visit.CurrentFullHandRecordCount);
                for (int index = 0; index < 10; index++)
                {
                    Run(visit, adapter,
                        AbstainAttempt(wordA, "capture_f32_defer_" + index, 6));
                    if (index < 9) Assert.IsTrue(visit.Retry());
                }
                Assert.IsTrue(visit.LeaveRoundIncomplete());
                Assert.AreEqual(wordB, visit.CurrentWord);
                Assert.IsTrue(visit.IsFullHandWordDeferred(wordA));
            }

            using (var resumed = new PetRobotSignPracticeSession(
                new FakeAdapter(), new RecordingRepository(), binding, Store(binding)))
            {
                Assert.AreEqual(wordA, resumed.CurrentWord,
                    "a fresh session offers the deferred word again");
                Assert.IsFalse(resumed.IsFullHandWordDeferred(wordA));
                Assert.AreEqual(1, resumed.CurrentFullHandRecordCount,
                    "the records collected earlier are retained");
                Assert.AreEqual(0, resumed.CurrentFullHandAttemptCount);
                Assert.IsTrue(resumed.PrepareAutoAttempt());
            }
        }

        // ---- the full-hand quest ------------------------------------------------

        [Test]
        public void TheFullHandQuestSessionRunsWithoutAnyLegacyHybridSession()
        {
            string wordA, wordB;
            FullHandEnrollmentBinding binding;
            CollectFullRound(out wordA, out wordB, out binding, true);
            FullHandParticipantCandidateStore store = Store(binding);

            Accepted accepted = AcceptedCapture(
                string.Equals(wordA, WordForOrder(false), StringComparison.Ordinal)
                    ? Direct() : Swapped(), 20, "capture_f28_quest");
            Assert.AreEqual(wordA, accepted.Word);

            var adapter = new FakeAdapter();
            var repository = new RecordingRepository();
            using (var session = new PetRobotSignPracticeSession(
                adapter, repository, new PetRobotGreetingQuestToken(), binding, store,
                null, wordA))
            {
                Assert.IsTrue(session.QuestMode);
                Assert.IsFalse(session.FullHandRecordCollectionActive,
                    "the quest route collects nothing");
                Assert.AreEqual(0L, session.QuestSessionGeneration,
                    "there is no legacy frozen hybrid session on this route");

                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.AreEqual(TrustedHybridOutcomeKind.StrictSuccess,
                                session.QuestAuthoritativeOutcome);
                Assert.IsTrue(session.QuestCompleted);
                Assert.IsNull(session.LegacyHybridDiagnosticOutcome);
                Assert.AreEqual(0, repository.SaveCalls);
                Assert.AreEqual(store.TotalCount, Store(binding).TotalCount,
                    "the quest attempt changes no stored record");
            }
        }

        [Test]
        public void AQuestAttemptWhoseEvidenceDoesNotMatchTheBindingNeverGrantsTheItem()
        {
            string wordA = WordForOrder(false);
            // the session is bound to one model identity while the evidence was scored under
            // another: an asserted StrictSuccess must not be enough
            var foreignBinding = new FullHandEnrollmentBinding(Participant, "model_f28_foreign",
                GateConfigId, new[] { wordA }, GateAuthorityId);
            FullHandParticipantCandidateStore foreignStore =
                FullHandParticipantCandidateStore.Open(root, Participant, foreignBinding);
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_foreign");
            Assert.AreEqual(wordA, accepted.Word);

            var adapter = new FakeAdapter();
            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), new PetRobotGreetingQuestToken(),
                foreignBinding, foreignStore, null, wordA))
            {
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.IsFalse(session.QuestCompleted,
                    "evidence from another model identity may not complete the quest");
                Assert.AreEqual(TrustedHybridOutcomeKind.TechnicalFailure,
                                session.QuestAuthoritativeOutcome);
            }

            // a legacy-shaped attempt on the same bound quest is a technical failure too
            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), new PetRobotGreetingQuestToken(),
                foreignBinding, foreignStore, null, wordA))
            {
                GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                    PersonalLiveProfileMode.MultiPrototypeCandidate,
                    PersonalLiveCaptureState.Completed, true, wordA, wordA,
                    true, true, true, true, true);
                Run(session, adapter, new PracticeRecognitionAttempt(
                    "capture_f28_quest_legacy", wordA, wordA,
                    PersonalLiveCaptureState.Completed, outcome, null, null,
                    PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(), "profile",
                    0.5d, null, null,
                    new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty }));
                Assert.IsFalse(session.QuestCompleted);
                Assert.AreEqual(TrustedHybridOutcomeKind.TechnicalFailure,
                                session.QuestAuthoritativeOutcome);
            }
        }

        [Test]
        public void AQuestAttemptWrappingAnEarlierCaptureNeverGrantsTheItem()
        {
            string wordA = WordForOrder(false);
            FullHandEnrollmentBinding binding = Binding(wordA, OtherWord(wordA));
            FullHandParticipantCandidateStore store = Store(binding);
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_earlier_practice");

            var adapter = new FakeAdapter();
            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), new PetRobotGreetingQuestToken(),
                binding, store, null, wordA))
            {
                // the accepted evidence and its decision both name the earlier capture, while
                // the attempt announces a new one: the three identities disagree
                Run(session, adapter, AttemptFor(accepted, "capture_f28_new_npc_attempt"));
                Assert.IsFalse(session.QuestCompleted,
                    "an earlier capture wrapped as a new attempt may not complete the quest");
                Assert.AreEqual(TrustedHybridOutcomeKind.TechnicalFailure,
                                session.QuestAuthoritativeOutcome);
            }

            // the SAME identities, all agreeing, still complete the quest
            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), new PetRobotGreetingQuestToken(),
                binding, store, null, wordA))
            {
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.IsTrue(session.QuestCompleted);
            }

            // and an ordinary abstention keeps its own honest verdict; the identity check
            // applies to the item-granting path only
            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), new PetRobotGreetingQuestToken(),
                binding, store, null, wordA))
            {
                const string abstainId = "capture_f28_quest_abstain";
                FullHandGameCaptureResult abstained =
                    Processor(new[] { Greeting, ThankYou }, RejectAllGates())
                        .Process(abstainId, wordA, RawCapture(6));
                FullHandGameplayDecision abstention =
                    FullHandGameplayEvaluator.Decide(abstained, wordA, abstainId, true);
                Assert.AreEqual(FullHandGameplayDecisionKind.NotDistinguished, abstention.Kind);
                Run(session, adapter, FullHandGameplayAttemptFactory.BuildAttempt(
                    abstention, abstainId, PersonalLiveCaptureState.Failed, null,
                    "f28_fake_profile", null, null, abstained));
                Assert.AreEqual(TrustedHybridOutcomeKind.NotDistinguishedRetry,
                                session.QuestAuthoritativeOutcome,
                    "an abstention stays an abstention and is not relabelled technical");
                Assert.IsFalse(session.QuestCompleted);
            }
        }

        [Test]
        public void AQuestWordOutsideTheVocabularyAndAMismatchedPinningAreRefused()
        {
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            FullHandParticipantCandidateStore store = Store(binding);
            Assert.Throws<InvalidOperationException>(() => new PetRobotSignPracticeSession(
                new FakeAdapter(), new RecordingRepository(), new PetRobotGreetingQuestToken(),
                binding, store, null, "LOVE"));

            FullHandEnrollmentBinding otherPinning = Binding(Greeting, ThankYou);
            Assert.Throws<InvalidOperationException>(() => new PetRobotSignPracticeSession(
                new FakeAdapter(), new RecordingRepository(), otherPinning, store));
        }

        // ---- off-mode and honest learner copy -----------------------------------

        [Test]
        public void TheOffModeSessionKeepsItsPreviousThreeWordBehaviour()
        {
            Accepted accepted = AcceptedCapture(Direct(), 6, "capture_f28_offmode");
            var adapter = new FakeAdapter();
            var repository = new RecordingRepository();
            using (var session = new PetRobotSignPracticeSession(adapter, repository))
            {
                Assert.AreEqual(3, session.Words.Count);
                Assert.IsFalse(session.FullHandRecordCollectionActive);
                Run(session, adapter, AttemptFor(accepted, accepted.CaptureId));
                Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess,
                                session.LastLearningOutcome.Kind);
                Assert.AreEqual(CandidateStorageStatus.NotRequested,
                                session.LastStorageResult.Status);
                Assert.AreEqual(FullHandGameplayFeedback.EnrollmentPending,
                                session.LastStorageResult.Message);
                Assert.AreEqual(0, repository.SaveCalls);
                Assert.AreEqual(0, session.CurrentFullHandRecordCount);
            }
            Assert.IsFalse(Directory.Exists(RecordDirectory()),
                "the off-mode route creates no record store at all");
        }

        /// <summary>Drives the REAL story flow with the REAL session view.</summary>
        static MoaStoryPresentation SessionStory(PetRobotSignPracticeSession session,
                                                 MoaStoryFeedbackKind feedback)
        {
            var flow = new MoaStoryGuidedPracticeFlow();
            flow.ConfigureVocabulary(session.Words);
            flow.Reset(session.CurrentWord);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.IsTrue(flow.AdvanceStory());
            Assert.IsTrue(flow.AdvanceStory());
            flow.Synchronize(session.CurrentWord, PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true,
                RawMediaPipeNeutralAutoSegmenterState.Inactive, true,
                false, PetRobotPersonalizationEnrollmentState.Pending, 0, 0, 0, false,
                true, session.FullHandRecordView);
            Assert.IsTrue(flow.ConfirmAutoAttemptAccepted());
            flow.Synchronize(session.CurrentWord, PetRobotPracticeState.Feedback, feedback,
                session.CanContinueWithNeedsPractice, true,
                RawMediaPipeNeutralAutoSegmenterState.Inactive, true,
                false, session.CurrentEnrollmentState, 0, 1, 0, false,
                true, session.FullHandRecordView);
            Assert.AreEqual(MoaStoryBeat.Feedback, flow.Beat);
            return flow.Current;
        }

        [Test]
        public void TheLearnerCopyReportsRecordProgressWithoutAnyPersonalizationClaim()
        {
            string wordA = WordForOrder(false);
            string wordB = WordForOrder(true);
            FullHandEnrollmentBinding binding = Binding(wordA, wordB);
            FullHandParticipantCandidateStore store = Store(binding);
            var adapter = new FakeAdapter();
            FullHandGameCaptureProcessor processor = Direct();

            using (var session = new PetRobotSignPracticeSession(
                adapter, new RecordingRepository(), binding, store))
            {
                Run(session, adapter,
                    AttemptFor(AcceptedCapture(processor, 6, "capture_f28_copy_0"),
                               "capture_f28_copy_0"));
                MoaStoryPresentation partial = SessionStory(session,
                    MoaStoryFeedbackKind.StrictSuccess);
                StringAssert.Contains("연습 기록 1/3", partial.BodyText);
                Assert.AreEqual(MoaStoryAction.RetryEnrollment, partial.Action,
                    "an uncollected word is practised again, not advanced");

                Assert.IsTrue(session.Retry());
                Run(session, adapter,
                    AttemptFor(AcceptedCapture(processor, 7, "capture_f28_copy_1"),
                               "capture_f28_copy_1"));
                Assert.IsTrue(session.Retry());
                Run(session, adapter,
                    AttemptFor(AcceptedCapture(processor, 8, "capture_f28_copy_2"),
                               "capture_f28_copy_2"));
                MoaStoryPresentation collected = SessionStory(session,
                    MoaStoryFeedbackKind.StrictSuccess);
                StringAssert.Contains(FullHandEnrollmentCopy.Collected, collected.BodyText);
                Assert.AreEqual(MoaStoryAction.ContinueToNextWord, collected.Action);
                Assert.AreEqual("다음 표현", collected.ActionLabel,
                    "the first word of a two-word round is not the last");

                foreach (string body in new[] { partial.BodyText, collected.BodyText })
                    foreach (string forbidden in new[] { "개인화 모델 완성", "개인화 등록", "표본",
                                                         "GREETING", "THANK_YOU", "hybrid",
                                                         "common_only", "wis_full_hand",
                                                         "MODEL_GATE_ACCEPTED" })
                        StringAssert.DoesNotContain(forbidden, body);
            }
        }

        [Test]
        public void AWordThisBuildCannotPresentIsRefusedInsteadOfMarkedComplete()
        {
            Assert.IsTrue(MoaStoryGuidedPracticeFlow.HasPresentationFor(Greeting));
            Assert.IsTrue(MoaStoryGuidedPracticeFlow.HasPresentationFor(ThankYou));
            Assert.IsFalse(MoaStoryGuidedPracticeFlow.HasPresentationFor("FUTURE_WORD"));

            var flow = new MoaStoryGuidedPracticeFlow();
            Assert.Throws<ArgumentException>(
                () => flow.ConfigureVocabulary(new[] { Greeting, "FUTURE_WORD" }));
            Assert.Throws<ArgumentException>(() => flow.ConfigureVocabulary(new string[0]));
            Assert.AreEqual(3, flow.Words.Count, "the refusal leaves the round untouched");
        }

        [Test]
        public void TheOverviewNamesOnlyTheWordsOfTheCurrentRound()
        {
            var flow = new MoaStoryGuidedPracticeFlow();
            flow.ConfigureVocabulary(new[] { Greeting, ThankYou });
            flow.Reset(Greeting);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.LessonOverview, flow.Beat);
            StringAssert.DoesNotContain("사랑", flow.Current.BodyText);
            StringAssert.DoesNotContain("세 표현", flow.Current.BodyText);
        }

        [Test]
        public void TheBindingPinsIdentityAndRefusesAMalformedVocabulary()
        {
            FullHandEnrollmentBinding binding = Binding(Greeting, ThankYou);
            Assert.AreEqual(3, binding.RequiredRecordings);
            Assert.IsTrue(binding.Supports(Greeting));
            Assert.IsFalse(binding.Supports("LOVE"));
            Assert.IsTrue(binding.IsLastWord(ThankYou));
            Assert.AreEqual(GateAuthorityId, binding.AuthorityGateId);
            Assert.AreEqual(GateConfigId + "_" + FullHandMethods.Hybrid,
                new FullHandEnrollmentBinding(Participant, ModelId, GateConfigId,
                    new[] { Greeting }).AuthorityGateId,
                "the default authority gate id follows the production gate loader");

            Assert.Throws<ArgumentException>(() => new FullHandEnrollmentBinding(
                Participant, ModelId, GateConfigId, new[] { Greeting, Greeting }));
            Assert.Throws<ArgumentException>(() => new FullHandEnrollmentBinding(
                Participant, ModelId, GateConfigId, new[] { "../escape" }));
            Assert.Throws<ArgumentException>(() => new FullHandEnrollmentBinding(
                " ", ModelId, GateConfigId, new[] { Greeting }));
        }
    }
}
