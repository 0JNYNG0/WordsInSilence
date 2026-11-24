// F24 - the bounded, READ-ONLY loader that connects an explicitly reviewed CURRENT-participant
// reference packet to the existing F20 pipeline through the F21 seam.
//
// This file adds NO geometry, NO classifier, NO threshold, NO fitted number, NO model and NO
// identity of its own. It contains no embedded participant id, no embedded path, no search
// path, no directory walk, no download and no write of any kind: it never creates a directory
// and never modifies a file. It reads exactly the packet the operator names, exactly the
// members that packet names, and nothing else.
//
// What it is NOT: it is not enrollment, not approval, not labelling and not quality assessment.
// The packet's labels and its selection of members are the product of an EXPLICIT LOCAL REVIEW
// that happens OUTSIDE this loader; the caller-supplied packet hash is the trust boundary for
// that review. A loaded reference is a RESEARCH COMPARISON input only. Nothing here proves the
// captured sign is well formed KSL, nothing here decides which real samples are correct, and no
// personalized improvement is claimed anywhere - earlier measurements showed that adding
// personal references can raise false accepts.
//
// The frozen F22 capture evidence is never edited or re-interpreted: a member trace must still
// declare quality_label UNASSESSED, eligible_for_enrollment false, its declared target as a
// REQUESTED expression and explicitly NOT ground truth. The reviewed label lives in the packet,
// beside the evidence, never inside it - and it is never taken from top1, from the requested
// target, from a score or from a threshold. No model score is recomputed to load a reference.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    /// <summary>
    /// Safe, non-secret refusal codes. None of these ever carries a path, a participant
    /// identity, a label or any capture content, so they are safe to surface in diagnostics.
    /// </summary>
    public static class FullHandReferencePacketCodes
    {
        /// <summary>Exactly one of the packet path and the expected packet hash was supplied.</summary>
        public const string ConfigurationIncomplete = "REFERENCES_CONFIGURATION_INCOMPLETE";
        public const string ArgumentInvalid = "REFERENCES_ARGUMENT_INVALID";
        public const string PathRejected = "REFERENCES_PATH_REJECTED";
        public const string PacketUnreadable = "REFERENCES_PACKET_UNREADABLE";
        public const string PacketTooLarge = "REFERENCES_PACKET_TOO_LARGE";
        public const string PacketHashMismatch = "REFERENCES_PACKET_HASH_MISMATCH";
        public const string PacketMalformed = "REFERENCES_PACKET_MALFORMED";
        public const string IdentityMismatch = "REFERENCES_IDENTITY_MISMATCH";
        public const string LabelUnsupported = "REFERENCES_LABEL_UNSUPPORTED";
        public const string MemberNameRejected = "REFERENCES_MEMBER_NAME_REJECTED";
        public const string MemberMissing = "REFERENCES_MEMBER_MISSING";
        public const string MemberTooLarge = "REFERENCES_MEMBER_TOO_LARGE";
        public const string MemberHashMismatch = "REFERENCES_MEMBER_HASH_MISMATCH";
        public const string TraceMalformed = "REFERENCES_TRACE_MALFORMED";
        /// <summary>A technical refusal is not evidence of a capture and is never a reference.</summary>
        public const string TraceNotCaptureOutcome = "REFERENCES_TRACE_NOT_CAPTURE_OUTCOME";
        /// <summary>The frozen capture evidence no longer reads as unassessed, non-enrollable.</summary>
        public const string TraceNotUnassessed = "REFERENCES_TRACE_NOT_UNASSESSED";
        /// <summary>The record's own frame count is not the frame count of its capture bytes.</summary>
        public const string FrameCountMismatch = "REFERENCES_FRAME_COUNT_MISMATCH";
        public const string BinaryMalformed = "REFERENCES_BINARY_MALFORMED";
        public const string BudgetExceeded = "REFERENCES_BUDGET_EXCEEDED";
        public const string FeaturesUnusable = "REFERENCES_FEATURES_UNUSABLE";
        public const string DigestMismatch = "REFERENCES_DIGEST_MISMATCH";
    }

    /// <summary>
    /// The reviewed current-participant reference packet. One static entry point; no instance
    /// state, no cache and no mutable configuration. The set it returns is an immutable
    /// snapshot for the lifetime of one controller: an update requires deliberate
    /// reconstruction, never an in-place edit of an already loaded set.
    /// </summary>
    public static class FullHandParticipantReferencePacket
    {
        public const string Schema = "wis_full_hand_participant_reference_packet_v1";
        public const string ReviewScope = "RESEARCH_COMPARISON_ONLY";
        public const string QualityBasis =
            "EXPLICIT_REVIEWED_SELECTION_NOT_AUTOMATIC_QUALITY_CERTIFICATION";

        public const string TraceSchema = "wis_full_hand_capture_trace_v1";
        public const string CaptureOutcomeKind = "capture_outcome";
        public const string UnassessedLabel = "UNASSESSED";
        public const string ManifestExtension = ".manifest.json";
        public const string BinaryExtension = ".bin";

        public const int MaxReferences = 64;
        public const int MaxPacketBytes = 256 * 1024;
        public const int MaxManifestBytes = 1024 * 1024;
        public const int MaxBinaryBytes = 16 * 1024 * 1024;
        public const long MaxTotalBinaryBytes = 64L * 1024L * 1024L;
        public const int MaxNameLength = 160;

        // the exact existing WISFHOB1 layout; nothing here is a new format
        const string BinaryMagic = "WISFHOB1";
        const int BinaryVersion = 1;
        const int BinaryPoints = 46;
        const int BinaryDims = 3;
        const int BinaryHeaderBytes = 24;              // 8 magic + 4 * int32
        // per frame: 8 timestamp + 46*3*8 coordinates + 3 * 46 mask bytes = 8 + 46*27
        const int BinaryPerFrameBytes = 8 + BinaryPoints * 27;

        // ---- entry point ---------------------------------------------------------

        /// <summary>
        /// Loads the reviewed packet, or returns the EMPTY set for the current participant.
        ///
        /// Both <paramref name="packetPath"/> and <paramref name="expectedPacketSha256"/>
        /// absent or blank is the default: the empty set is returned without a single file
        /// system operation, so the unconfigured behaviour is byte-for-byte unchanged. Exactly
        /// one of the two supplied is a configuration mistake, never a silent empty fallback.
        /// Every other failure throws a typed <see cref="FullHandFormatException"/>.
        /// </summary>
        public static FullHandParticipantReferenceSet LoadOrEmpty(
            string packetPath, string expectedPacketSha256, string expectedParticipantId,
            string expectedCommonModelId, string expectedGateConfigId, string[] vocabulary)
        {
            bool hasPath = !string.IsNullOrWhiteSpace(packetPath);
            bool hasHash = !string.IsNullOrWhiteSpace(expectedPacketSha256);

            RequireKnown(expectedParticipantId, "owner");
            RequireKnown(expectedCommonModelId, "model");
            RequireKnown(expectedGateConfigId, "gate");
            string[] words = RequireVocabulary(vocabulary);

            if (!hasPath && !hasHash)
                // the default path: no IO of any kind, and nothing personal is loaded
                return FullHandParticipantReferenceSet.Empty(expectedParticipantId);
            if (hasPath != hasHash)
                throw Refuse(FullHandReferencePacketCodes.ConfigurationIncomplete,
                    "a reference packet needs both its path and its expected hash");

            string wantedPacketHash = RequireSha256(expectedPacketSha256,
                FullHandReferencePacketCodes.ArgumentInvalid, "the expected packet hash");

            RequireSafeAbsolutePath(packetPath);
            RequireSafeExistingFile(packetPath, FullHandReferencePacketCodes.PacketUnreadable);
            byte[] packetBytes = ReadBoundedFile(packetPath, MaxPacketBytes,
                FullHandReferencePacketCodes.PacketTooLarge,
                FullHandReferencePacketCodes.PacketUnreadable);
            if (!string.Equals(Sha256Hex(packetBytes), wantedPacketHash, StringComparison.Ordinal))
                throw Refuse(FullHandReferencePacketCodes.PacketHashMismatch,
                    "the packet bytes are not the reviewed bytes");

            PacketJson packet = ParseJson<PacketJson>(packetBytes,
                FullHandReferencePacketCodes.PacketMalformed, "the packet");
            RequireEqual(packet.schema, Schema, FullHandReferencePacketCodes.PacketMalformed,
                "the packet schema");
            RequireEqual(packet.review_scope, ReviewScope,
                FullHandReferencePacketCodes.PacketMalformed, "the declared review scope");
            RequireEqual(packet.quality_basis, QualityBasis,
                FullHandReferencePacketCodes.PacketMalformed, "the declared quality basis");

            // identity is settled BEFORE a single member is touched
            RequireEqual(packet.participant_id, expectedParticipantId,
                FullHandReferencePacketCodes.IdentityMismatch, "the packet owner");
            RequireEqual(packet.common_model_id, expectedCommonModelId,
                FullHandReferencePacketCodes.IdentityMismatch, "the packet model");
            RequireEqual(packet.gate_config_id, expectedGateConfigId,
                FullHandReferencePacketCodes.IdentityMismatch, "the packet gate configuration");

            ReferenceJson[] entries = packet.references;
            if (entries == null || entries.Length == 0)
                throw Refuse(FullHandReferencePacketCodes.PacketMalformed,
                    "the packet declares no reviewed reference");
            if (entries.Length > MaxReferences)
                throw Refuse(FullHandReferencePacketCodes.BudgetExceeded,
                    "the packet declares more references than the accepted limit");

            string directory = DirectoryOf(packetPath);
            var samples = new FullHandSample[entries.Length];
            long totalBinaryBytes = 0;
            for (int index = 0; index < entries.Length; index++)
                samples[index] = LoadReference(entries[index], directory, expectedParticipantId,
                    expectedCommonModelId, expectedGateConfigId, words, ref totalBinaryBytes);

            // the existing duplicate id / duplicate source hash / duplicate feature digest
            // guards live in F20 and are relied on, not re-implemented here
            return new FullHandParticipantReferenceSet(expectedParticipantId, samples);
        }

        // ---- one reviewed member -------------------------------------------------

        static FullHandSample LoadReference(ReferenceJson entry, string directory,
            string participantId, string modelId, string gateConfigId, string[] vocabulary,
            ref long totalBinaryBytes)
        {
            if (entry == null)
                throw Refuse(FullHandReferencePacketCodes.PacketMalformed, "an absent entry");

            // the reviewed label. It is INPUT: it is not read from top1, from the requested
            // target, from a score or from any threshold anywhere in this file.
            string label = entry.label;
            if (string.IsNullOrEmpty(label) || IndexOf(vocabulary, label) < 0)
                throw Refuse(FullHandReferencePacketCodes.LabelUnsupported,
                    "a reviewed label is not part of the loaded vocabulary");

            string manifestName = entry.trace_manifest_file;
            if (!IsPlainSafeName(manifestName, ManifestExtension))
                throw Refuse(FullHandReferencePacketCodes.MemberNameRejected,
                    "a member is not a plain completed manifest name");
            string wantedManifestHash = RequireSha256(entry.trace_manifest_sha256,
                FullHandReferencePacketCodes.PacketMalformed, "a member hash");

            string manifestPath = Path.Combine(directory, manifestName);
            RequireSafeExistingFile(manifestPath, FullHandReferencePacketCodes.MemberMissing);
            byte[] manifestBytes = ReadBoundedFile(manifestPath, MaxManifestBytes,
                FullHandReferencePacketCodes.MemberTooLarge,
                FullHandReferencePacketCodes.MemberMissing);
            if (!string.Equals(Sha256Hex(manifestBytes), wantedManifestHash,
                               StringComparison.Ordinal))
                throw Refuse(FullHandReferencePacketCodes.MemberHashMismatch,
                    "a member manifest is not the reviewed manifest");

            string manifestText = Utf8(manifestBytes, FullHandReferencePacketCodes.TraceMalformed);
            TraceJson trace = ParseJson<TraceJson>(manifestBytes,
                FullHandReferencePacketCodes.TraceMalformed, "a member manifest");

            RequireEqual(trace.schema, TraceSchema, FullHandReferencePacketCodes.TraceMalformed,
                "a member schema");
            // a technical refusal is preserved as a refusal and can never become a reference
            RequireEqual(trace.record_kind, CaptureOutcomeKind,
                FullHandReferencePacketCodes.TraceNotCaptureOutcome, "a member record kind");

            RequireEqual(trace.participant_id, participantId,
                FullHandReferencePacketCodes.IdentityMismatch, "a member owner");
            RequireEqual(trace.common_model_id, modelId,
                FullHandReferencePacketCodes.IdentityMismatch, "a member model");
            RequireEqual(trace.gate_config_id, gateConfigId,
                FullHandReferencePacketCodes.IdentityMismatch, "a member gate configuration");

            // the ROOT scope of the record, read with a scoped scanner so that a supported word
            // that merely LOOKS like a key - or a nested field of the same name - can neither
            // satisfy nor break a required guarantee
            TraceRootScope root = TraceRootScope.Read(manifestText);

            // the frozen evidence must still read exactly as F22 wrote it. Selecting a sample
            // for research comparison is a SEPARATE explicit review, never an edit of this.
            root.RequireLiteral(TraceRootScope.QualityLabelKey, "\"" + UnassessedLabel + "\"");
            root.RequireLiteral(TraceRootScope.EligibleKey, "false");
            root.RequireLiteral(TraceRootScope.RequestedExpressionKey, "true");
            root.RequireLiteral(TraceRootScope.GroundTruthKey, "false");
            root.RequireLiteral(TraceRootScope.ScoresPresentKey, "true");

            int declaredFrames = root.RequirePositiveInteger(TraceRootScope.FrameCountKey);

            string sampleId = trace.capture_session_id;
            if (string.IsNullOrEmpty(sampleId) ||
                !string.Equals(sampleId, trace.query_sample_id, StringComparison.Ordinal))
                throw Refuse(FullHandReferencePacketCodes.TraceMalformed,
                    "a member does not identify one single capture");
            string wantedDigest = RequireSha256(trace.query_feature_digest,
                FullHandReferencePacketCodes.TraceMalformed, "a member feature digest");

            SourceBinaryJson source = trace.source_binary;
            if (source == null)
                throw Refuse(FullHandReferencePacketCodes.TraceMalformed,
                    "a member declares no source binary");
            if (!IsPlainSafeName(source.file, BinaryExtension))
                throw Refuse(FullHandReferencePacketCodes.MemberNameRejected,
                    "a member binary is not a plain retained capture name");
            string wantedBinaryHash = RequireSha256(source.sha256,
                FullHandReferencePacketCodes.TraceMalformed, "a member binary hash");
            string pipelineHash = RequireSha256(source.source_sha256_from_pipeline,
                FullHandReferencePacketCodes.TraceMalformed, "a member pipeline hash");
            if (!string.Equals(wantedBinaryHash, pipelineHash, StringComparison.Ordinal))
                throw Refuse(FullHandReferencePacketCodes.TraceMalformed,
                    "a member binary hash disagrees with the pipeline hash");

            // read from the source_binary object ONLY: an unrelated nested byte_count is not it
            int declaredBytes = root.SourceBinaryByteCount();
            if (declaredBytes > MaxBinaryBytes)
                throw Refuse(FullHandReferencePacketCodes.MemberTooLarge,
                    "a member binary is larger than the accepted limit");
            checked { totalBinaryBytes += declaredBytes; }
            if (totalBinaryBytes > MaxTotalBinaryBytes)
                throw Refuse(FullHandReferencePacketCodes.BudgetExceeded,
                    "the packet declares more capture bytes than the accepted limit");

            string binaryPath = Path.Combine(directory, source.file);
            RequireSafeExistingFile(binaryPath, FullHandReferencePacketCodes.MemberMissing);
            byte[] binary = ReadBoundedFile(binaryPath, declaredBytes,
                FullHandReferencePacketCodes.MemberTooLarge,
                FullHandReferencePacketCodes.MemberMissing);
            if (binary.Length != declaredBytes)
                throw Refuse(FullHandReferencePacketCodes.MemberHashMismatch,
                    "a member binary is not the declared length");
            if (!string.Equals(Sha256Hex(binary), wantedBinaryHash, StringComparison.Ordinal))
                throw Refuse(FullHandReferencePacketCodes.MemberHashMismatch,
                    "a member binary is not the recorded capture");

            double[] coordinates;
            bool[] finalMask;
            int frames;
            ParseCaptureBinary(binary, out coordinates, out finalMask, out frames);
            // the record's own frame count and the decoded capture must agree EXACTLY, and they
            // are compared before any feature is built
            if (frames != declaredFrames)
                throw Refuse(FullHandReferencePacketCodes.FrameCountMismatch,
                    "a member declares a different frame count than its capture bytes");

            FullHandFeatureSequence sequence;
            try { sequence = FullHandPairGeometry.BuildFeatures(coordinates, finalMask, frames); }
            catch (FullHandFormatException exception)
            {
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                    "a member capture was refused by the approved geometry (" +
                    exception.Code + ")");
            }
            if (!sequence.Usable)
                throw Refuse(FullHandReferencePacketCodes.FeaturesUnusable,
                    "a member capture carries no usable reference shape");

            // the ONE binding: these exact bytes produce exactly the features the frozen record
            // was scored from. Nothing is re-scored to establish that.
            if (!string.Equals(FullHandFeatureDigest.Compute(sequence), wantedDigest,
                               StringComparison.Ordinal))
                throw Refuse(FullHandReferencePacketCodes.DigestMismatch,
                    "a member does not reproduce its recorded feature digest");

            return new FullHandSample(sampleId, participantId, wantedBinaryHash, label, sequence);
        }

        // ---- the existing WISFHOB1 layout ---------------------------------------

        /// <summary>
        /// F28, ADDITIVE and narrowly exposed. Decodes RETAINED capture bytes with the exact
        /// reader below and the exact approved geometry, and returns the resulting feature
        /// sequence. Every malformed, truncated, trailing-byte, non-finite and mask refusal
        /// still applies unchanged.
        ///
        /// It is NOT a packet load: no manifest is read, no reviewed selection is implied, no
        /// label, quality or enrollment eligibility is granted, and no reference is registered.
        /// The reviewed-packet route above remains the only way a participant sample can ever
        /// become a scored reference.
        /// </summary>
        internal static FullHandFeatureSequence DecodeRetainedCaptureFeatures(
            byte[] binary, out int frames)
        {
            if (binary == null)
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed, "an absent capture");
            double[] coordinates;
            bool[] finalMask;
            ParseCaptureBinary(binary, out coordinates, out finalMask, out frames);
            return FullHandPairGeometry.BuildFeatures(coordinates, finalMask, frames);
        }

        /// <summary>
        /// Reads the exact existing capture layout. Every length is checked BEFORE any array is
        /// allocated, the frame count is bounded by the byte limit under checked arithmetic, and
        /// a trailing or truncated byte is a refusal, never a best effort.
        /// </summary>
        static void ParseCaptureBinary(byte[] binary, out double[] coordinates,
                                       out bool[] finalMask, out int frames)
        {
            if (binary.Length < BinaryHeaderBytes)
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed, "a truncated header");
            for (int index = 0; index < BinaryMagic.Length; index++)
                if (binary[index] != (byte)BinaryMagic[index])
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed, "an unknown magic");
            if (ReadInt32(binary, 8) != BinaryVersion ||
                ReadInt32(binary, 16) != BinaryPoints ||
                ReadInt32(binary, 20) != BinaryDims)
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                    "an unsupported capture shape");

            frames = ReadInt32(binary, 12);
            int maxFrames = (MaxBinaryBytes - BinaryHeaderBytes) / BinaryPerFrameBytes;
            if (frames <= 0 || frames > maxFrames)
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                    "an unsupported frame count");

            long expected;
            checked { expected = BinaryHeaderBytes + (long)frames * BinaryPerFrameBytes; }
            if (binary.LongLength != expected)
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                    "the capture is truncated or carries trailing bytes");

            int slots = frames * BinaryPoints;
            coordinates = new double[slots * BinaryDims];
            finalMask = new bool[slots];

            int at = BinaryHeaderBytes;
            double previous = double.NegativeInfinity;
            for (int frame = 0; frame < frames; frame++)
            {
                double timestamp = ReadDouble(binary, at);
                at += 8;
                if (double.IsNaN(timestamp) || double.IsInfinity(timestamp))
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                        "a non-finite timestamp");
                if (!(timestamp > previous))
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                        "the timestamps are not strictly increasing");
                previous = timestamp;
            }
            for (int index = 0; index < coordinates.Length; index++)
            {
                coordinates[index] = ReadDouble(binary, at);
                at += 8;
            }

            int finalAt = at;
            int originalAt = at + slots;
            int interpolatedAt = at + slots * 2;
            for (int slot = 0; slot < slots; slot++)
            {
                bool isFinal = ReadFlag(binary, finalAt + slot);
                bool isOriginal = ReadFlag(binary, originalAt + slot);
                bool isInterpolated = ReadFlag(binary, interpolatedAt + slot);
                if (isOriginal && isInterpolated)
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                        "a point is both original and interpolated");
                // the reverse is NOT required: an original-valid point may legitimately be
                // dropped by the final normalized transform, so the masks need not be equal
                if (isFinal && !isOriginal && !isInterpolated)
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                        "a final-valid point has no source evidence");
                finalMask[slot] = isFinal;
                if (!isFinal) continue;      // a masked coordinate is ignored, never read as data
                double x = coordinates[slot * BinaryDims];
                double y = coordinates[slot * BinaryDims + 1];
                double z = coordinates[slot * BinaryDims + 2];
                if (double.IsNaN(x) || double.IsInfinity(x) ||
                    double.IsNaN(y) || double.IsInfinity(y))
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                        "a valid point has a non-finite coordinate");
                if (z != 0.0)
                    throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                        "a valid point has a non-zero z; this contract is masked 2D");
            }
        }

        static bool ReadFlag(byte[] binary, int at)
        {
            byte value = binary[at];
            if (value > 1)
                throw Refuse(FullHandReferencePacketCodes.BinaryMalformed,
                    "a mask byte is neither 0 nor 1");
            return value == 1;
        }

        static int ReadInt32(byte[] binary, int at)
        {
            return binary[at] | (binary[at + 1] << 8) | (binary[at + 2] << 16) |
                   (binary[at + 3] << 24);
        }

        static double ReadDouble(byte[] binary, int at)
        {
            long bits = 0;
            for (int shift = 7; shift >= 0; shift--)
                bits = (bits << 8) | binary[at + shift];
            return BitConverter.Int64BitsToDouble(bits);
        }

        // ---- typed json ----------------------------------------------------------

        [Serializable]
        sealed class ReferenceJson
        {
            public string trace_manifest_file;
            public string trace_manifest_sha256;
            public string label;
        }

        [Serializable]
        sealed class PacketJson
        {
            public string schema;
            public string participant_id;
            public string common_model_id;
            public string gate_config_id;
            public string review_scope;
            public string quality_basis;
            public ReferenceJson[] references;
        }

        [Serializable]
        sealed class SourceBinaryJson
        {
            public string file;
            public string sha256;
            public string source_sha256_from_pipeline;
        }

        /// <summary>Only the fields this loader actually needs. Nothing else is interpreted.</summary>
        [Serializable]
        sealed class TraceJson
        {
            public string schema;
            public string record_kind;
            public string capture_session_id;
            public string participant_id;
            public string query_sample_id;
            public string query_feature_digest;
            public string common_model_id;
            public string gate_config_id;
            public SourceBinaryJson source_binary;
        }

        static T ParseJson<T>(byte[] bytes, string code, string what) where T : class
        {
            string text = Utf8(bytes, code);
            T parsed;
            try { parsed = JsonUtility.FromJson<T>(text); }
            catch (Exception)
            {
                throw Refuse(code, what + " is not valid JSON");
            }
            if (parsed == null) throw Refuse(code, what + " is not a JSON object");
            return parsed;
        }

        static string Utf8(byte[] bytes, string code)
        {
            try { return new UTF8Encoding(false, true).GetString(StripBom(bytes)); }
            catch (Exception) { throw Refuse(code, "the document is not valid UTF-8"); }
        }

        static byte[] StripBom(byte[] bytes)
        {
            if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
            {
                var trimmed = new byte[bytes.Length - 3];
                Array.Copy(bytes, 3, trimmed, 0, trimmed.Length);
                return trimmed;
            }
            return bytes;
        }

        // ---- the scoped root scope of one capture record -------------------------

        /// <summary>
        /// The handful of guarantees JsonUtility cannot read safely, taken from the ROOT object
        /// of the record and from the <c>source_binary</c> object only.
        ///
        /// JsonUtility cannot tell a missing bool from <c>false</c> or a missing number from
        /// zero, so these are read as EXACT literals from the document itself. A plain text
        /// search is not good enough either: a supported word may literally be spelled
        /// "quality_label", and a nested member of the same name must never satisfy a root
        /// guarantee. So this walks the object structure - strings, escapes, objects and arrays
        /// - and only ever looks at the IMMEDIATE members of the scope it was given. It is not a
        /// general parser, it has no dependency, and it interprets nothing: it returns verbatim
        /// value tokens. An absent key and a key declared twice in the same scope are refusals.
        /// </summary>
        sealed class TraceRootScope
        {
            internal const string QualityLabelKey = "quality_label";
            internal const string EligibleKey = "eligible_for_enrollment";
            internal const string RequestedExpressionKey =
                "declared_target_is_requested_expression";
            internal const string GroundTruthKey = "declared_target_is_ground_truth";
            internal const string ScoresPresentKey = "scores_present";
            internal const string FrameCountKey = "frame_count";
            internal const string SourceBinaryKey = "source_binary";
            internal const string ByteCountKey = "byte_count";

            static readonly string[] RootKeys =
            {
                QualityLabelKey, EligibleKey, RequestedExpressionKey, GroundTruthKey,
                ScoresPresentKey, FrameCountKey, SourceBinaryKey
            };

            readonly string text;
            readonly int[] starts;
            readonly int[] ends;

            TraceRootScope(string text, int[] starts, int[] ends)
            {
                this.text = text;
                this.starts = starts;
                this.ends = ends;
            }

            internal static TraceRootScope Read(string text)
            {
                int at = SkipWhitespace(text, 0);
                if (at >= text.Length || text[at] != '{')
                    throw Refuse(FullHandReferencePacketCodes.TraceMalformed,
                        "a member manifest is not a JSON object");
                var starts = new int[RootKeys.Length];
                var ends = new int[RootKeys.Length];
                ScanObject(text, at, RootKeys, starts, ends);
                return new TraceRootScope(text, starts, ends);
            }

            string Token(string key)
            {
                for (int slot = 0; slot < RootKeys.Length; slot++)
                {
                    if (!string.Equals(RootKeys[slot], key, StringComparison.Ordinal)) continue;
                    return starts[slot] < 0
                        ? null : text.Substring(starts[slot], ends[slot] - starts[slot]);
                }
                return null;
            }

            /// <summary>Exact literal semantics. An absent key is a refusal, never a default.</summary>
            internal void RequireLiteral(string key, string expected)
            {
                if (!string.Equals(Token(key), expected, StringComparison.Ordinal))
                    throw Refuse(FullHandReferencePacketCodes.TraceNotUnassessed,
                        "a member no longer declares its frozen capture guarantee");
            }

            internal int RequirePositiveInteger(string key)
            {
                return ParsePositiveInteger(Token(key));
            }

            /// <summary>The byte count of the declared source binary, from that object alone.</summary>
            internal int SourceBinaryByteCount()
            {
                int slot = Array.IndexOf(RootKeys, SourceBinaryKey);
                if (starts[slot] < 0 || text[starts[slot]] != '{')
                    throw Refuse(FullHandReferencePacketCodes.TraceMalformed,
                        "a member declares no source binary object");
                var keys = new[] { ByteCountKey };
                var innerStarts = new int[1];
                var innerEnds = new int[1];
                ScanObject(text, starts[slot], keys, innerStarts, innerEnds);
                return ParsePositiveInteger(innerStarts[0] < 0
                    ? null : text.Substring(innerStarts[0], innerEnds[0] - innerStarts[0]));
            }

            static int ParsePositiveInteger(string token)
            {
                int value;
                if (token == null || token.Length > 16 ||
                    !int.TryParse(token, NumberStyles.None, CultureInfo.InvariantCulture,
                                  out value) || value <= 0)
                    throw Refuse(FullHandReferencePacketCodes.TraceMalformed,
                        "a member declares no positive count where one is required");
                return value;
            }

            // ---- the scanner ----------------------------------------------------

            /// <summary>
            /// Records the verbatim value span of each wanted key among the IMMEDIATE members of
            /// the object starting at <paramref name="objectStart"/>. Nested members are walked
            /// over, never matched. A wanted key declared twice in this scope is a refusal.
            /// </summary>
            static void ScanObject(string text, int objectStart, string[] wanted,
                                   int[] starts, int[] ends)
            {
                for (int slot = 0; slot < wanted.Length; slot++) { starts[slot] = -1; ends[slot] = -1; }
                int index = SkipWhitespace(text, objectStart + 1);
                if (index < text.Length && text[index] == '}') return;
                while (true)
                {
                    index = SkipWhitespace(text, index);
                    if (index >= text.Length || text[index] != '"')
                        throw Malformed("a member name was expected");
                    int keyStart = index + 1;
                    index = SkipString(text, index);
                    int keyEnd = index - 1;
                    index = SkipWhitespace(text, index);
                    if (index >= text.Length || text[index] != ':')
                        throw Malformed("a member separator was expected");
                    index = SkipWhitespace(text, index + 1);
                    int valueStart = index;
                    index = SkipValue(text, index);
                    int slot = MatchKey(text, keyStart, keyEnd, wanted);
                    if (slot >= 0)
                    {
                        if (starts[slot] >= 0)
                            throw Malformed("a required member is declared more than once");
                        starts[slot] = valueStart;
                        ends[slot] = index;
                    }
                    index = SkipWhitespace(text, index);
                    if (index < text.Length && text[index] == ',') { index++; continue; }
                    if (index < text.Length && text[index] == '}') return;
                    throw Malformed("a malformed object member");
                }
            }

            /// <summary>An escaped key never matches: an unexpected spelling fails closed.</summary>
            static int MatchKey(string text, int start, int end, string[] wanted)
            {
                int length = end - start;
                for (int index = start; index < end; index++)
                    if (text[index] == '\\') return -1;
                for (int slot = 0; slot < wanted.Length; slot++)
                {
                    if (wanted[slot].Length != length) continue;
                    bool same = true;
                    for (int offset = 0; offset < length && same; offset++)
                        if (text[start + offset] != wanted[slot][offset]) same = false;
                    if (same) return slot;
                }
                return -1;
            }

            static int SkipWhitespace(string text, int index)
            {
                while (index < text.Length && char.IsWhiteSpace(text[index])) index++;
                return index;
            }

            /// <summary>From the opening quote to just past the closing quote.</summary>
            static int SkipString(string text, int index)
            {
                index++;
                while (index < text.Length)
                {
                    char c = text[index];
                    if (c == '\\') { index += 2; continue; }
                    if (c == '"') return index + 1;
                    index++;
                }
                throw Malformed("an unterminated string");
            }

            static int SkipValue(string text, int index)
            {
                if (index >= text.Length) throw Malformed("a value was expected");
                char c = text[index];
                if (c == '"') return SkipString(text, index);
                if (c == '{' || c == '[')
                {
                    int depth = 0;
                    while (index < text.Length)
                    {
                        char current = text[index];
                        if (current == '"') { index = SkipString(text, index); continue; }
                        if (current == '{' || current == '[') depth++;
                        else if (current == '}' || current == ']')
                        {
                            depth--;
                            if (depth == 0) return index + 1;
                        }
                        index++;
                    }
                    throw Malformed("an unterminated object or array");
                }
                int start = index;
                while (index < text.Length)
                {
                    char current = text[index];
                    if (current == ',' || current == '}' || current == ']' ||
                        char.IsWhiteSpace(current)) break;
                    index++;
                }
                if (index == start) throw Malformed("an empty value");
                return index;
            }

            static FullHandFormatException Malformed(string detail)
            {
                return Refuse(FullHandReferencePacketCodes.TraceMalformed, detail);
            }
        }

        // ---- paths ---------------------------------------------------------------

        /// <summary>
        /// A fully qualified LOCAL path with no traversal segment, no UNC or device prefix and
        /// no alternate data stream. Nothing is normalized away and nothing is repaired.
        /// </summary>
        static void RequireSafeAbsolutePath(string path)
        {
            bool ok;
            try { ok = Path.IsPathFullyQualified(path); }
            catch (Exception) { ok = false; }
            if (!ok)
                throw Refuse(FullHandReferencePacketCodes.PathRejected,
                    "the packet path is not a fully qualified local path");
            if (path.StartsWith("\\\\", StringComparison.Ordinal) ||
                path.StartsWith("//", StringComparison.Ordinal) ||
                path.IndexOf("\\\\?\\", StringComparison.Ordinal) >= 0 ||
                path.IndexOf("\\\\.\\", StringComparison.Ordinal) >= 0)
                throw Refuse(FullHandReferencePacketCodes.PathRejected,
                    "a UNC or device path is not a local packet");
            if (path.IndexOf(':', 2) >= 0)
                throw Refuse(FullHandReferencePacketCodes.PathRejected,
                    "the packet path carries a stream or device separator");
            foreach (string segment in path.Split('/', '\\'))
                if (segment == "." || segment == "..")
                    throw Refuse(FullHandReferencePacketCodes.PathRejected,
                        "the packet path carries a traversal segment");
        }

        /// <summary>
        /// The file must exist, and neither it nor ANY ancestor directory up to the root may be
        /// a reparse point: a junction or symbolic link anywhere along the accessed path could
        /// redirect the read outside the reviewed directory, so the whole chain is inspected.
        /// This only inspects; it never creates a directory and never writes.
        /// </summary>
        static void RequireSafeExistingFile(string path, string missingCode)
        {
            try
            {
                if (!File.Exists(path))
                    throw Refuse(missingCode, "a required file is absent");
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                    throw Refuse(FullHandReferencePacketCodes.PathRejected,
                        "a reparse point is not accepted evidence");
                var directory = new DirectoryInfo(DirectoryOf(path));
                while (directory != null)
                {
                    if (!directory.Exists)
                        throw Refuse(FullHandReferencePacketCodes.PathRejected,
                            "the accessed path crosses a directory that is not present");
                    if ((directory.Attributes & FileAttributes.ReparsePoint) != 0)
                        throw Refuse(FullHandReferencePacketCodes.PathRejected,
                            "the accessed path crosses a reparse point");
                    directory = directory.Parent;
                }
            }
            catch (FullHandFormatException) { throw; }
            catch (Exception)
            {
                throw Refuse(FullHandReferencePacketCodes.PathRejected,
                    "a required file could not be inspected");
            }
        }

        static string DirectoryOf(string path)
        {
            string directory;
            try { directory = Path.GetDirectoryName(path); }
            catch (Exception) { directory = null; }
            if (string.IsNullOrEmpty(directory))
                throw Refuse(FullHandReferencePacketCodes.PathRejected,
                    "the packet has no containing directory");
            return directory;
        }

        /// <summary>
        /// A plain F22 output name and nothing else: no directory separator, no traversal, no
        /// stream, no leading dot and the exact completed suffix. A staging name can never
        /// match, so incomplete evidence is never read as a finished record.
        /// </summary>
        static bool IsPlainSafeName(string name, string suffix)
        {
            if (string.IsNullOrEmpty(name) || name.Length > MaxNameLength) return false;
            if (name.Length <= suffix.Length) return false;
            if (!name.EndsWith(suffix, StringComparison.Ordinal)) return false;
            if (name[0] == '.' || name.IndexOf("..", StringComparison.Ordinal) >= 0) return false;
            for (int index = 0; index < name.Length; index++)
            {
                char c = name[index];
                bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
                if (!allowed) return false;
            }
            return true;
        }

        /// <summary>
        /// Reads one file through ONE opened stream: the length is taken from that same stream
        /// and checked before a single byte is allocated, exactly that many bytes are read, and
        /// an early end or an unexpected trailing byte is a refusal. Nothing is opened for
        /// writing and nothing is created.
        /// </summary>
        static byte[] ReadBoundedFile(string path, long limit, string tooLargeCode,
                                      string unreadableCode)
        {
            try
            {
                using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                                                   FileShare.Read))
                {
                    long length = stream.Length;
                    if (length <= 0)
                        throw Refuse(unreadableCode, "a required file is empty");
                    if (length > limit)
                        throw Refuse(tooLargeCode, "a file is larger than the accepted limit");
                    var buffer = new byte[(int)length];
                    int at = 0;
                    while (at < buffer.Length)
                    {
                        int read = stream.Read(buffer, at, buffer.Length - at);
                        if (read <= 0)
                            throw Refuse(unreadableCode,
                                "a required file ended before its declared length");
                        at += read;
                    }
                    if (stream.ReadByte() != -1)
                        throw Refuse(unreadableCode,
                            "a required file carries unexpected trailing bytes");
                    return buffer;
                }
            }
            catch (FullHandFormatException) { throw; }
            catch (Exception)
            {
                throw Refuse(unreadableCode, "a required file could not be read");
            }
        }

        // ---- small helpers -------------------------------------------------------

        static void RequireKnown(string value, string what)
        {
            if (string.IsNullOrWhiteSpace(value))
                throw Refuse(FullHandReferencePacketCodes.ArgumentInvalid,
                    "the current " + what + " identity is required");
        }

        static string[] RequireVocabulary(string[] vocabulary)
        {
            if (vocabulary == null || vocabulary.Length == 0)
                throw Refuse(FullHandReferencePacketCodes.ArgumentInvalid,
                    "a known non-empty vocabulary is required");
            var copy = (string[])vocabulary.Clone();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            for (int index = 0; index < copy.Length; index++)
            {
                if (string.IsNullOrWhiteSpace(copy[index]) || !seen.Add(copy[index]))
                    throw Refuse(FullHandReferencePacketCodes.ArgumentInvalid,
                        "the vocabulary must be non-empty and unique");
            }
            return copy;
        }

        static int IndexOf(string[] values, string value)
        {
            for (int index = 0; index < values.Length; index++)
                if (string.Equals(values[index], value, StringComparison.Ordinal)) return index;
            return -1;
        }

        static void RequireEqual(string actual, string expected, string code, string what)
        {
            if (!string.Equals(actual, expected, StringComparison.Ordinal))
                throw Refuse(code, what + " does not match the current configuration");
        }

        static string RequireSha256(string value, string code, string what)
        {
            if (string.IsNullOrEmpty(value) || value.Length != 64)
                throw Refuse(code, what + " is not 64 hex characters");
            var text = new StringBuilder(64);
            for (int index = 0; index < 64; index++)
            {
                char c = value[index];
                bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                           (c >= 'A' && c <= 'F');
                if (!hex) throw Refuse(code, what + " is not 64 hex characters");
                text.Append(char.ToLowerInvariant(c));
            }
            return text.ToString();
        }

        public static string Sha256Hex(byte[] data)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(data);
                var text = new StringBuilder(digest.Length * 2);
                for (int index = 0; index < digest.Length; index++)
                    text.Append(digest[index].ToString("x2"));
                return text.ToString();
            }
        }

        /// <summary>Codes and short generic phrases only: never a path, an id or capture data.</summary>
        static FullHandFormatException Refuse(string code, string detail)
        {
            return new FullHandFormatException(code, detail);
        }
    }
}
