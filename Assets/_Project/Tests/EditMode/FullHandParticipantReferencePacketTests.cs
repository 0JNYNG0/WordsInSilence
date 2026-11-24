// F24 EditMode tests for the reviewed current-participant reference loader.
//
// Everything here is synthetic and in memory except ONE isolated temporary directory that each
// test creates and deletes itself. No camera, no persistentDataPath, no project file, no private
// learner data, no model file, no real profile, no fixture, no network and no application
// persistence. The bank, the gates and the captures are constructed by the test, so no
// production cutoff and no fitted number is asserted anywhere.
//
// The packet members are GENUINE F22 output: a real WISFHOB1 binary from
// FullHandObservationSession.BuildBinary plus the manifest FullHandCaptureTraceWriter actually
// wrote. The reviewed label is chosen by the test as a stand-in for an explicit local review; it
// is deliberately NOT the requested target, so nothing here can be read as auto-labelling or as
// evidence that a synthetic capture is a well formed sign.
//
// Every negative test that edits a member REFRESHES the reviewed hashes afterwards, so each
// refusal is a refusal of the evidence itself and not merely of a stale hash.
//
// Not covered here, by instruction: reparse points. Creating a junction or a symbolic link needs
// privilege, so the ancestor check is left for the operator to probe independently against a
// safe temporary junction after this compiles.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandParticipantReferencePacketTests
    {
        const int SourceFrames = 4;
        const string Participant = "participant_a";
        const string OtherParticipant = "participant_b";
        const string ModelId = "model_test_v1";
        const string GateConfigId = "config_test";
        static readonly string[] Words = { "kiwi", "zebra" };   // arbitrary labels, no word rules

        // the requested target of the capture, and the SEPARATE reviewed label of the packet
        const string RequestedTarget = "kiwi";
        const string ReviewedLabel = "zebra";

        /// <summary>
        /// Arbitrary supported values that happen to collide with manifest key names, or that
        /// need JSON escaping. They are stand-ins for ordinary vocabulary entries; no game word
        /// is hardcoded and nothing here depends on their meaning.
        /// </summary>
        static readonly string[] AwkwardWords =
        {   // the last entry carries a non-ASCII letter, built in code so this file stays ASCII
            "quality_label", "byte_count", "a\"b\\c", "caf" + (char)0xE9 + "_word"
        };

        string root;

        [SetUp]
        public void CreateIsolatedDirectory()
        {
            root = Path.Combine(Path.GetTempPath(), "wis_f24_refs_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
        }

        [TearDown]
        public void RemoveIsolatedDirectory()
        {
            try { if (Directory.Exists(root)) Directory.Delete(root, true); }
            catch (Exception) { /* a leftover temp directory is not a test failure */ }
        }

        // ---- synthetic pipeline material ---------------------------------------

        static void Arrays(double seed, out double[] coordinates, out bool[] mask)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            coordinates = new double[SourceFrames * points * dims];
            mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = seed + point * 0.13 + frame * 0.05;
                    coordinates[slot * dims + 1] = seed * 0.5 + point * 0.07 - frame * 0.03;
                    coordinates[slot * dims + 2] = 0.0;
                }
        }

        static FullHandFeatureSequence Sequence(double seed)
        {
            double[] coordinates; bool[] mask;
            Arrays(seed, out coordinates, out mask);
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, SourceFrames);
        }

        static FullHandCommonCapture Capture(double seed)
        {
            double[] coordinates; bool[] mask;
            Arrays(seed, out coordinates, out mask);
            return new FullHandCommonCapture(SourceFrames, coordinates, mask, mask,
                new bool[mask.Length], mask.Length, 0, 640, 480, "test_convention");
        }

        static double[] Timestamps()
        {
            var values = new double[SourceFrames];
            for (int index = 0; index < SourceFrames; index++) values[index] = index * 0.04;
            return values;
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

        static FullHandInMemoryCommonBank Bank(string[] words)
        {
            var ids = new string[4];
            var labels = new int[4];
            var sequences = new FullHandFeatureSequence[4];
            for (int index = 0; index < 4; index++)
            {
                ids[index] = "common_" + index;
                labels[index] = index % words.Length;
                sequences[index] = Sequence(1.0 + index * 0.4);
            }
            return new FullHandInMemoryCommonBank(ModelId, words, ids, labels, sequences,
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
        }

        static FullHandGateConfig Gates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.AtQualityThreshold(GateConfigId + "_" + index,
                    FullHandMethods.All[index], 0.5);
            return new FullHandGateConfig(GateConfigId, gates);
        }

        static FullHandRecognitionPipeline Pipeline(FullHandParticipantReferenceSet references)
        {
            return Pipeline(references, Words);
        }

        static FullHandRecognitionPipeline Pipeline(FullHandParticipantReferenceSet references,
                                                    string[] words)
        {
            return new FullHandRecognitionPipeline(Bank(words), Gates(), Participant, references);
        }

        // ---- genuine F22 evidence ------------------------------------------------

        string RecordCapture(double seed, string captureId)
        {
            return RecordCapture(seed, captureId, Words, RequestedTarget);
        }

        /// <summary>
        /// Produces one real completed F22 pair in the isolated directory and returns the
        /// manifest file name the writer chose.
        /// </summary>
        string RecordCapture(double seed, string captureId, string[] words, string target)
        {
            FullHandCommonCapture capture = Capture(seed);
            byte[] binary = FullHandObservationSession.BuildBinary(capture, Timestamps());
            string sourceSha = FullHandCaptureTraceWriter.Sha256Hex(binary);
            FullHandRecognitionOutcome outcome = Pipeline(
                FullHandParticipantReferenceSet.Empty(Participant), words)
                .Evaluate(capture, captureId, sourceSha, target);
            FullHandGameCaptureResult result = FullHandGameCaptureResult.ForOutcome(
                captureId, SourceFrames, outcome, sourceSha, binary);

            var before = new HashSet<string>(Directory.GetFiles(root, "*.manifest.json"),
                                             StringComparer.Ordinal);
            string code = new FullHandCaptureTraceWriter(root).Record(result);
            Assert.AreEqual(FullHandTraceCodes.None, code, "the synthetic evidence must be written");
            foreach (string path in Directory.GetFiles(root, "*.manifest.json"))
                if (!before.Contains(path)) return Path.GetFileName(path);
            Assert.Fail("no manifest was produced");
            return null;
        }

        static string Hex(byte[] bytes) { return FullHandCaptureTraceWriter.Sha256Hex(bytes); }

        static string HashOf(string path) { return Hex(File.ReadAllBytes(path)); }

        static void Quote(StringBuilder text, string value)
        {
            for (int index = 0; index < value.Length; index++)
            {
                char c = value[index];
                if (c == '"' || c == '\\') text.Append('\\');
                text.Append(c);
            }
        }

        string Packet(string participantId, string modelId, string gateConfigId, string padding,
                      params string[][] entries)
        {
            var text = new StringBuilder();
            text.Append("{\"schema\":\"").Append(FullHandParticipantReferencePacket.Schema);
            text.Append("\",\"participant_id\":\"").Append(participantId);
            text.Append("\",\"common_model_id\":\"").Append(modelId);
            text.Append("\",\"gate_config_id\":\"").Append(gateConfigId);
            text.Append("\",\"review_scope\":\"").Append(FullHandParticipantReferencePacket.ReviewScope);
            text.Append("\",\"quality_basis\":\"").Append(FullHandParticipantReferencePacket.QualityBasis);
            if (padding != null) text.Append("\",\"reviewer_note\":\"").Append(padding);
            text.Append("\",\"references\":[");
            for (int index = 0; index < entries.Length; index++)
            {
                if (index > 0) text.Append(',');
                text.Append("{\"trace_manifest_file\":\"").Append(entries[index][0]);
                text.Append("\",\"trace_manifest_sha256\":\"").Append(entries[index][1]);
                text.Append("\",\"label\":\"");
                Quote(text, entries[index][2]);
                text.Append("\"}");
            }
            text.Append("]}");
            string path = Path.Combine(root, "packet_" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllBytes(path, new UTF8Encoding(false).GetBytes(text.ToString()));
            return path;
        }

        /// <summary>One reviewed entry naming a manifest that is on disk right now.</summary>
        string[] Entry(string manifestName, string label)
        {
            return new[] { manifestName, HashOf(Path.Combine(root, manifestName)), label };
        }

        string Load(string packetPath, out FullHandParticipantReferenceSet set)
        {
            return Load(packetPath, Words, out set);
        }

        string Load(string packetPath, string[] words, out FullHandParticipantReferenceSet set)
        {
            string sha = HashOf(packetPath);
            set = FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, sha, Participant,
                ModelId, GateConfigId, words);
            return sha;
        }

        static string CodeOf(TestDelegate action)
        {
            var exception = Assert.Throws<FullHandFormatException>(action);
            return exception.Code;
        }

        /// <summary>
        /// Rewrites a manifest after a member file was deliberately changed. Callers rehash
        /// afterwards through <see cref="Entry"/>, so the packet stays internally consistent and
        /// each refusal is about the evidence, not about a stale hash.
        /// </summary>
        void Rewrite(string manifestName, string from, string to)
        {
            string path = Path.Combine(root, manifestName);
            string text = File.ReadAllText(path);
            Assert.IsTrue(text.Contains(from), "the manifest must contain the patched fragment");
            File.WriteAllBytes(path, new UTF8Encoding(false).GetBytes(text.Replace(from, to)));
        }

        static string BinaryNameOf(string manifestText)
        {
            int at = manifestText.IndexOf("\"file\":\"", StringComparison.Ordinal) + 8;
            int end = manifestText.IndexOf('"', at);
            return manifestText.Substring(at, end - at);
        }

        // ---- the default path ----------------------------------------------------

        [Test]
        public void NoConfiguredPacketReturnsTheEmptySetAndTouchesNothing()
        {
            string absent = Path.Combine(root, "never_created");
            FullHandParticipantReferenceSet set =
                FullHandParticipantReferencePacket.LoadOrEmpty(null, null, Participant, ModelId,
                    GateConfigId, Words);
            Assert.AreEqual(0, set.Count);
            Assert.AreEqual(Participant, set.ParticipantId);
            Assert.IsFalse(Directory.Exists(absent), "the loader must never create a directory");
            Assert.AreEqual(0, Directory.GetFiles(root).Length);

            set = FullHandParticipantReferencePacket.LoadOrEmpty("   ", "  ", Participant, ModelId,
                GateConfigId, Words);
            Assert.AreEqual(0, set.Count);
        }

        [Test]
        public void HalfAConfigurationIsRefusedAndNeverSilentlyEmpty()
        {
            string manifestName = RecordCapture(5.0, "capture_half");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            string sha = HashOf(packetPath);

            Assert.AreEqual(FullHandReferencePacketCodes.ConfigurationIncomplete,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, null,
                    Participant, ModelId, GateConfigId, Words)));
            Assert.AreEqual(FullHandReferencePacketCodes.ConfigurationIncomplete,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(null, sha,
                    Participant, ModelId, GateConfigId, Words)));
        }

        [Test]
        public void UnknownOwnerOrVocabularyIsRefusedBeforeAnyIo()
        {
            Assert.AreEqual(FullHandReferencePacketCodes.ArgumentInvalid,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(null, null, "",
                    ModelId, GateConfigId, Words)));
            Assert.AreEqual(FullHandReferencePacketCodes.ArgumentInvalid,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(null, null,
                    Participant, ModelId, GateConfigId, new[] { "kiwi", "kiwi" })));
            Assert.AreEqual(FullHandReferencePacketCodes.ArgumentInvalid,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(null, null,
                    Participant, ModelId, GateConfigId, new string[0])));
        }

        // ---- the successful path --------------------------------------------------

        [Test]
        public void ReviewedPacketLoadsExactlyTheReviewedReference()
        {
            string manifestName = RecordCapture(5.0, "capture_one");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));

            FullHandParticipantReferenceSet set;
            Load(packetPath, out set);

            Assert.AreEqual(1, set.Count);
            FullHandSample sample = set.At(0);
            Assert.AreEqual("capture_one", sample.SampleId);
            Assert.AreEqual(Participant, sample.ParticipantId);
            // the label is the REVIEWED input, not the requested target of the capture
            Assert.AreEqual(ReviewedLabel, sample.Label);
            Assert.AreNotEqual(RequestedTarget, sample.Label);
            Assert.AreEqual(FullHandFeatureDigest.Compute(Sequence(5.0)), sample.FeatureDigest);
        }

        [Test]
        public void LoadedReferenceIsExcludedFromItsOwnQueryAndScoredOnceForOthers()
        {
            string manifestName = RecordCapture(5.0, "capture_self");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Load(packetPath, out set);

            FullHandRecognitionPipeline personal = Pipeline(set);
            Assert.AreEqual(1, personal.CurrentReferenceCount);

            // the same input under a different id: F20's own alias exclusion drops it, so the
            // two banks are identical and the single evaluation is reused, not repeated
            FullHandRecognitionOutcome self = personal.Evaluate(Capture(5.0), "query_self",
                null, RequestedTarget);
            Assert.AreEqual(1, self.ScoringPasses);
            Assert.IsTrue(self.ReusedCommonEvaluation);
            Assert.AreEqual(1, self.After.ExcludedParticipantReferenceCount);
            Assert.AreEqual(0, self.After.ParticipantReferenceCount);

            FullHandRecognitionOutcome common = Pipeline(
                FullHandParticipantReferenceSet.Empty(Participant))
                .Evaluate(Capture(5.0), "query_self", null, RequestedTarget);
            for (int index = 0; index < FullHandMethods.All.Length; index++)
            {
                string method = FullHandMethods.All[index];
                Assert.AreEqual(common.After.Decision(method).Decision,
                                self.After.Decision(method).Decision,
                                "an excluded self reference must not change the decision");
                Assert.AreEqual(common.After.Decision(method).Score,
                                self.After.Decision(method).Score, 0.0);
            }

            // a different capture does see the reference, and only then is there a second pass
            FullHandRecognitionOutcome other = personal.Evaluate(Capture(7.0), "query_other",
                null, RequestedTarget);
            Assert.AreEqual(2, other.ScoringPasses);
            Assert.AreEqual(1, other.After.ParticipantReferenceCount);
            Assert.AreEqual(0, other.Before.ParticipantReferenceCount);
        }

        [Test]
        public void LoadingReadsAndMutatesNothing()
        {
            string manifestName = RecordCapture(5.0, "capture_frozen");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            var before = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string path in Directory.GetFiles(root)) before[path] = HashOf(path);

            FullHandParticipantReferenceSet set;
            Load(packetPath, out set);

            string[] after = Directory.GetFiles(root);
            Assert.AreEqual(before.Count, after.Length, "no file may be added or removed");
            foreach (string path in after)
                Assert.AreEqual(before[path], HashOf(path), "no byte on disk may change");
        }

        [Test]
        public void AnAlreadyLoadedSetIsUnaffectedByLaterFileEdits()
        {
            string manifestName = RecordCapture(5.0, "capture_snapshot");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Load(packetPath, out set);
            string digest = set.At(0).FeatureDigest;

            foreach (string path in Directory.GetFiles(root)) File.Delete(path);

            // the loaded set is an immutable snapshot: a later edit cannot reach it, and the
            // only way to change it is a deliberate reconstruction
            Assert.AreEqual(1, set.Count);
            Assert.AreEqual(digest, set.At(0).FeatureDigest);
            Assert.AreEqual(1, Pipeline(set).CurrentReferenceCount);
        }

        // ---- identity ------------------------------------------------------------

        [Test]
        public void PacketsForAnotherParticipantModelOrGateAreRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_identity");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            string sha = HashOf(packetPath);

            Assert.AreEqual(FullHandReferencePacketCodes.IdentityMismatch,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, sha,
                    OtherParticipant, ModelId, GateConfigId, Words)),
                "a packet from a previous participant is never reused after a reset");
            Assert.AreEqual(FullHandReferencePacketCodes.IdentityMismatch,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, sha,
                    Participant, "model_other_v1", GateConfigId, Words)));
            Assert.AreEqual(FullHandReferencePacketCodes.IdentityMismatch,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, sha,
                    Participant, ModelId, "config_other", Words)));
        }

        [Test]
        public void AMemberRecordedUnderAnotherIdentityIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_member_identity");
            Rewrite(manifestName, "\"participant_id\":\"" + Participant + "\"",
                    "\"participant_id\":\"" + OtherParticipant + "\"");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.IdentityMismatch,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void APacketWhoseBytesAreNotTheReviewedBytesIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_hash");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            string wrong = new string('a', 64);

            Assert.AreEqual(FullHandReferencePacketCodes.PacketHashMismatch,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, wrong,
                    Participant, ModelId, GateConfigId, Words)));
            Assert.AreEqual(FullHandReferencePacketCodes.ArgumentInvalid,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(packetPath, "abc",
                    Participant, ModelId, GateConfigId, Words)));
        }

        // ---- the evidence must still be evidence ---------------------------------

        [Test]
        public void ATamperedOrTruncatedCaptureIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_tamper");
            string binaryPath = Path.Combine(root,
                BinaryNameOf(File.ReadAllText(Path.Combine(root, manifestName))));
            byte[] original = File.ReadAllBytes(binaryPath);

            byte[] flipped = (byte[])original.Clone();
            flipped[original.Length / 2] ^= 0x01;
            File.WriteAllBytes(binaryPath, flipped);
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.MemberHashMismatch,
                CodeOf(() => Load(packetPath, out set)));

            var truncated = new byte[original.Length - 16];
            Array.Copy(original, truncated, truncated.Length);
            File.WriteAllBytes(binaryPath, truncated);
            Assert.AreEqual(FullHandReferencePacketCodes.MemberHashMismatch,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void AWrongShapeCaptureIsRefusedEvenWhenTheHashesWereRedone()
        {
            string manifestName = RecordCapture(5.0, "capture_shape");
            string manifestText = File.ReadAllText(Path.Combine(root, manifestName));
            string binaryPath = Path.Combine(root, BinaryNameOf(manifestText));
            byte[] binary = File.ReadAllBytes(binaryPath);
            string oldSha = Hex(binary);
            binary[16] = 45;                       // the point count is no longer 46
            File.WriteAllBytes(binaryPath, binary);
            Rewrite(manifestName, oldSha, Hex(binary));

            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.BinaryMalformed,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void ADifferentCaptureUnderTheSameRecordIsRefusedByTheFeatureDigest()
        {
            string manifestName = RecordCapture(5.0, "capture_digest");
            string manifestText = File.ReadAllText(Path.Combine(root, manifestName));
            string binaryPath = Path.Combine(root, BinaryNameOf(manifestText));
            string oldSha = HashOf(binaryPath);

            // a genuine, well formed capture of something else, with every hash redone so the
            // packet is byte-consistent: only the recorded feature digest still disagrees
            byte[] swapped = FullHandObservationSession.BuildBinary(Capture(7.0), Timestamps());
            File.WriteAllBytes(binaryPath, swapped);
            Rewrite(manifestName, oldSha, Hex(swapped));

            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.DigestMismatch,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void ATechnicalRefusalOrAnEditedGuaranteeIsNeverAReference()
        {
            string manifestName = RecordCapture(5.0, "capture_refusal");
            Rewrite(manifestName, "\"record_kind\":\"capture_outcome\"",
                    "\"record_kind\":\"technical_refusal\"");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.TraceNotCaptureOutcome,
                CodeOf(() => Load(packetPath, out set)));

            string second = RecordCapture(6.0, "capture_enrollable");
            Rewrite(second, "\"eligible_for_enrollment\":false",
                    "\"eligible_for_enrollment\":true");
            string secondPacket = Packet(Participant, ModelId, GateConfigId, null,
                Entry(second, ReviewedLabel));
            Assert.AreEqual(FullHandReferencePacketCodes.TraceNotUnassessed,
                CodeOf(() => Load(secondPacket, out set)));

            string third = RecordCapture(8.0, "capture_ground_truth");
            Rewrite(third, "\"declared_target_is_ground_truth\":false",
                    "\"declared_target_is_ground_truth\":true");
            string thirdPacket = Packet(Participant, ModelId, GateConfigId, null,
                Entry(third, ReviewedLabel));
            Assert.AreEqual(FullHandReferencePacketCodes.TraceNotUnassessed,
                CodeOf(() => Load(thirdPacket, out set)));
        }

        // ---- the scoped record scan ----------------------------------------------

        [Test]
        public void SupportedWordsThatLookLikeKeysOrNeedEscapingStillLoad()
        {
            // the vocabulary carries values spelled exactly like manifest keys, plus values that
            // must be JSON-escaped: they appear all over the recorded class scores, and none of
            // them may be mistaken for a root guarantee
            string label = AwkwardWords[2];
            string manifestName = RecordCapture(5.0, "capture_awkward", AwkwardWords,
                                                AwkwardWords[0]);
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, label));

            FullHandParticipantReferenceSet set;
            Load(packetPath, AwkwardWords, out set);
            Assert.AreEqual(1, set.Count);
            Assert.AreEqual(label, set.At(0).Label);
            Assert.AreNotEqual(AwkwardWords[0], set.At(0).Label);
            Assert.AreEqual(1, Pipeline(set, AwkwardWords).CurrentReferenceCount);
        }

        [Test]
        public void ARequiredGuaranteeDeclaredOnlyInANestedObjectIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_nested_flag");
            // removed from the root, then re-declared inside a nested object: a nested member of
            // the same name must never satisfy a root guarantee
            Rewrite(manifestName, ",\"eligible_for_enrollment\":false", "");
            Rewrite(manifestName, "\"banks\":{", "\"banks\":{\"eligible_for_enrollment\":false,");

            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.TraceNotUnassessed,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void ARootGuaranteeDeclaredTwiceIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_duplicate_key");
            Rewrite(manifestName, "{\"schema\":", "{\"scores_present\":true,\"schema\":");

            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.TraceMalformed,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void NestedCountsOfTheSameNameAreIgnored()
        {
            string manifestName = RecordCapture(5.0, "capture_nested_counts");
            // an unrelated nested byte_count and frame_count: the real ones are read from the
            // root and from the source_binary object only, so this changes nothing
            Rewrite(manifestName, "\"banks\":{",
                    "\"banks\":{\"byte_count\":999,\"frame_count\":99,");

            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Load(packetPath, out set);
            Assert.AreEqual(1, set.Count);
        }

        [Test]
        public void AMissingOrDisagreeingFrameCountIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_frames");
            string declared = ",\"frame_count\":" + SourceFrames;
            Rewrite(manifestName, declared, ",\"frame_count\":" + (SourceFrames + 1));
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.FrameCountMismatch,
                CodeOf(() => Load(packetPath, out set)),
                "the record's own frame count must equal the decoded capture");

            string second = RecordCapture(6.0, "capture_frames_absent");
            Rewrite(second, declared, "");
            string secondPacket = Packet(Participant, ModelId, GateConfigId, null,
                Entry(second, ReviewedLabel));
            Assert.AreEqual(FullHandReferencePacketCodes.TraceMalformed,
                CodeOf(() => Load(secondPacket, out set)),
                "an absent frame count is a refusal, never a zero");
        }

        // ---- names, selection and bounds -----------------------------------------

        [Test]
        public void AnUnsupportedReviewedLabelIsRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_label");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, "not_in_the_vocabulary"));
            FullHandParticipantReferenceSet set;
            Assert.AreEqual(FullHandReferencePacketCodes.LabelUnsupported,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void TheSameEvidenceUnderTwoNamesIsRefusedAsADuplicate()
        {
            string manifestName = RecordCapture(5.0, "capture_alias");
            string alias = "cap_000001_alias_copy" + FullHandCaptureTraceWriter.ManifestExtension;
            File.Copy(Path.Combine(root, manifestName), Path.Combine(root, alias));

            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel), Entry(alias, RequestedTarget));
            FullHandParticipantReferenceSet set;
            // F20's own duplicate id / source hash / feature digest guards do this work
            Assert.AreEqual(FullHandPipelineCodes.DuplicateSample,
                CodeOf(() => Load(packetPath, out set)));
        }

        [Test]
        public void TraversalStagingAndMissingMembersAreRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_names");
            FullHandParticipantReferenceSet set;

            string escaping = Packet(Participant, ModelId, GateConfigId, null,
                new[] { "sub/../" + manifestName, HashOf(Path.Combine(root, manifestName)),
                        ReviewedLabel });
            Assert.AreEqual(FullHandReferencePacketCodes.MemberNameRejected,
                CodeOf(() => Load(escaping, out set)));

            // an incomplete record wears the staging name and is never a finished record
            string staging = "cap_000001_partial" + FullHandCaptureTraceWriter.StagingExtension;
            File.Copy(Path.Combine(root, manifestName), Path.Combine(root, staging));
            string stagingPacket = Packet(Participant, ModelId, GateConfigId, null,
                new[] { staging, HashOf(Path.Combine(root, staging)), ReviewedLabel });
            Assert.AreEqual(FullHandReferencePacketCodes.MemberNameRejected,
                CodeOf(() => Load(stagingPacket, out set)));

            string missingBinary = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            File.Delete(Path.Combine(root,
                BinaryNameOf(File.ReadAllText(Path.Combine(root, manifestName)))));
            Assert.AreEqual(FullHandReferencePacketCodes.MemberMissing,
                CodeOf(() => Load(missingBinary, out set)));
        }

        [Test]
        public void ThePacketPathMustBeAPlainFullyQualifiedLocalFile()
        {
            string manifestName = RecordCapture(5.0, "capture_path");
            string packetPath = Packet(Participant, ModelId, GateConfigId, null,
                Entry(manifestName, ReviewedLabel));
            string sha = HashOf(packetPath);
            string traversal = Path.Combine(root, "sub", "..", Path.GetFileName(packetPath));

            Assert.AreEqual(FullHandReferencePacketCodes.PathRejected,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(traversal, sha,
                    Participant, ModelId, GateConfigId, Words)));
            Assert.AreEqual(FullHandReferencePacketCodes.PathRejected,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(
                    Path.GetFileName(packetPath), sha, Participant, ModelId, GateConfigId, Words)));
            Assert.AreEqual(FullHandReferencePacketCodes.PacketUnreadable,
                CodeOf(() => FullHandParticipantReferencePacket.LoadOrEmpty(
                    Path.Combine(root, "absent_packet.json"), sha, Participant, ModelId,
                    GateConfigId, Words)));
        }

        [Test]
        public void EmptyOversizedAndOvercountedPacketsAreRefused()
        {
            string manifestName = RecordCapture(5.0, "capture_bounds");
            string[] entry = Entry(manifestName, ReviewedLabel);
            FullHandParticipantReferenceSet set;

            string empty = Packet(Participant, ModelId, GateConfigId, null);
            Assert.AreEqual(FullHandReferencePacketCodes.PacketMalformed,
                CodeOf(() => Load(empty, out set)));

            var many = new string[FullHandParticipantReferencePacket.MaxReferences + 1][];
            for (int index = 0; index < many.Length; index++) many[index] = entry;
            string overcounted = Packet(Participant, ModelId, GateConfigId, null, many);
            Assert.AreEqual(FullHandReferencePacketCodes.BudgetExceeded,
                CodeOf(() => Load(overcounted, out set)));

            string oversized = Packet(Participant, ModelId, GateConfigId,
                new string('n', FullHandParticipantReferencePacket.MaxPacketBytes + 1), entry);
            Assert.AreEqual(FullHandReferencePacketCodes.PacketTooLarge,
                CodeOf(() => Load(oversized, out set)));
        }

        // ---- the off-by-default configuration seam --------------------------------

        [Test]
        public void TheConfigurationSeamStaysOffByDefault()
        {
            Assert.AreEqual("WIS_FULL_HAND_PIPELINE_REFERENCES",
                FullHandGameCaptureConfiguration.ReferencesVariable);
            Assert.AreEqual("WIS_FULL_HAND_PIPELINE_REFERENCES_SHA256",
                FullHandGameCaptureConfiguration.ReferencesSha256Variable);

            string previous = Environment.GetEnvironmentVariable(
                FullHandGameCaptureConfiguration.EnableVariable);
            try
            {
                Environment.SetEnvironmentVariable(
                    FullHandGameCaptureConfiguration.EnableVariable, null);
                // disabled means null: no file is read, no participant is touched and the
                // reference variables above are never consulted at all
                Assert.IsFalse(FullHandGameCaptureConfiguration.IsEnabled());
                Assert.IsNull(FullHandGameCaptureConfiguration.CreateProcessorOrNull());
            }
            finally
            {
                Environment.SetEnvironmentVariable(
                    FullHandGameCaptureConfiguration.EnableVariable, previous);
            }
        }
    }
}
