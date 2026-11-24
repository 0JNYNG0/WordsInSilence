// F28 - the LOCAL, bounded, per-participant store of full-hand practice recordings.
//
// What one record is: ONE capture of the current participant that the FROZEN common-only
// bank/gate ACCEPTED for the prompted, supported word, together with the exact WISFHOB1
// source bytes that produced it, its source hash, its feature digest and the model / gate /
// participant / capture identities that were pinned at the time.
//
// What one record is NOT: it is not user-verified, not expert-verified, not a certificate
// that a well formed Korean Sign Language sign was performed, and not a personalized
// reference. Its quality status is exactly MODEL_GATE_ACCEPTED. Nothing in this file loads a
// stored record back into the recognition pipeline: the gameplay authority stays the
// common-only "Before" bank, the "After" bank stays comparison-only, and the F24 reviewed
// packet remains the ONLY route by which a participant reference can ever be scored against.
//
// The hashes here establish that a stored record is the SAME bytes and the SAME features that
// were scored - they are an integrity check against accidental change, not a security or
// authenticity system, and they say nothing about sign correctness.
//
// This file adds NO geometry, NO classifier, NO threshold and NO fitted number. It writes only
// under its own directory, never touches the 16-point candidate files, the AIHub or the
// three-view material, and it fails CLOSED on every integrity doubt.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using WordsInSilence.Recognition.FullHandRuntime;

namespace WordsInSilence.Practice.FullHandEnrollment
{
    public enum FullHandEnrollmentSaveStatus
    {
        /// <summary>No save was attempted for this attempt. The default.</summary>
        NotRequested = 0,
        /// <summary>A new unique accepted recording was committed.</summary>
        Saved,
        /// <summary>The same capture / source / feature evidence is already stored.</summary>
        Duplicate,
        /// <summary>The evidence is not an accepted, supported, matching decision.</summary>
        NotAccepted,
        /// <summary>Identity, integrity or bounds refused this record.</summary>
        Refused,
        /// <summary>The write itself failed. No sample is consumed; the caller retries.</summary>
        StorageFailed,
    }

    public sealed class FullHandEnrollmentSaveResult
    {
        public FullHandEnrollmentSaveStatus Status { get; private set; }
        public string Word { get; private set; }
        public string FailureCode { get; private set; }
        public string FailureDetail { get; private set; }
        /// <summary>The count for the word AFTER this call, recomputed from the store.</summary>
        public int CountForWord { get; private set; }

        public bool Committed { get { return Status == FullHandEnrollmentSaveStatus.Saved; } }

        public FullHandEnrollmentSaveResult(FullHandEnrollmentSaveStatus status, string word,
                                            int countForWord, string failureCode,
                                            string failureDetail)
        {
            Status = status;
            Word = word ?? string.Empty;
            CountForWord = countForWord < 0 ? 0 : countForWord;
            FailureCode = failureCode ?? FullHandEnrollmentCodes.None;
            FailureDetail = failureDetail ?? string.Empty;
        }

        public static readonly FullHandEnrollmentSaveResult NotRequested =
            new FullHandEnrollmentSaveResult(FullHandEnrollmentSaveStatus.NotRequested,
                string.Empty, 0, FullHandEnrollmentCodes.None, string.Empty);
    }

    /// <summary>One committed, re-verified record. Immutable and read-only.</summary>
    public sealed class FullHandParticipantCandidateRecord
    {
        public string RecordId { get; private set; }
        public string Word { get; private set; }
        public string ParticipantId { get; private set; }
        public string ModelId { get; private set; }
        public string GateConfigId { get; private set; }
        public string GateId { get; private set; }
        public string CaptureSessionId { get; private set; }
        public string SourceSha256 { get; private set; }
        public string FeatureDigest { get; private set; }
        public int FrameCount { get; private set; }
        public int SourceByteCount { get; private set; }
        public string QualityStatus { get; private set; }
        public string QualityBasis { get; private set; }
        public DateTime SavedUtc { get; private set; }

        internal FullHandParticipantCandidateRecord(string recordId, string word,
            string participantId, string modelId, string gateConfigId, string gateId,
            string captureSessionId, string sourceSha256, string featureDigest, int frameCount,
            int sourceByteCount, string qualityStatus, string qualityBasis, DateTime savedUtc)
        {
            RecordId = recordId;
            Word = word;
            ParticipantId = participantId;
            ModelId = modelId;
            GateConfigId = gateConfigId;
            GateId = gateId;
            CaptureSessionId = captureSessionId;
            SourceSha256 = sourceSha256;
            FeatureDigest = featureDigest;
            FrameCount = frameCount;
            SourceByteCount = sourceByteCount;
            QualityStatus = qualityStatus;
            QualityBasis = qualityBasis;
            SavedUtc = savedUtc;
        }
    }

    /// <summary>
    /// The per-participant full-hand candidate store.
    ///
    /// Layout: &lt;CandidateRoot&gt;/&lt;learnerId&gt;/FullHandCandidatesV1/&lt;recordId&gt;.json
    /// beside &lt;recordId&gt;.bin. CandidateRoot is SHARED across participants, so the learner
    /// id segment is the whole separation: a StartNew reset allocates a different learner id
    /// and therefore addresses an empty directory, while every previously written file is left
    /// untouched.
    ///
    /// The record id is derived from a hash of the pinned identities plus the capture, source
    /// and feature identities. No capture-supplied or word-supplied text ever becomes part of
    /// a path.
    /// </summary>
    public sealed class FullHandParticipantCandidateStore
    {
        public const string DirectoryName = "FullHandCandidatesV1";
        public const string Schema = "wis_full_hand_participant_candidate_v1";
        /// <summary>The ONLY quality claim this store makes. Not user/expert verified.</summary>
        public const string QualityStatusModelGateAccepted = "MODEL_GATE_ACCEPTED";
        public const string QualityBasisLiteral =
            "FROZEN_MODEL_GATE_DECISION_NOT_USER_OR_EXPERT_VERIFIED";
        public const string RecordExtension = ".json";
        public const string BinaryExtension = ".bin";
        public const string StagingExtension = ".tmp";
        public const string SavedUtcFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'";
        public const string RecordIdPrefix = "fhc_";
        public const int RecordIdLength = 44;          // prefix + 40 lowercase hex

        // hard, sensible bounds. Nothing here is a recognition threshold.
        public const int MaxRecordsPerWord = 32;
        public const int MaxRecords = 256;
        public const int MaxFilesInspected = 2048;
        public const int MaxRecordBytes = 8 * 1024;
        public const int MaxSourceBytes = 16 * 1024 * 1024;
        public const long MaxTotalSourceBytes = 256L * 1024L * 1024L;
        public const int MaxLearnerIdLength = 96;

        static readonly string[] RequiredKeys =
        {
            "schema", "record_id", "participant_id", "model_id", "gate_config_id", "gate_id",
            "method", "bank_id", "word", "declared_target", "top1", "decision",
            "target_supported", "target_matches", "capture_session_id", "query_sample_id",
            "feature_digest", "source_sha256", "binary_file", "binary_sha256", "binary_bytes",
            "frame_count", "quality_status", "quality_basis", "saved_utc",
        };

        readonly FullHandEnrollmentBinding binding;
        readonly string candidateRoot;
        readonly Dictionary<string, List<FullHandParticipantCandidateRecord>> byWord =
            new Dictionary<string, List<FullHandParticipantCandidateRecord>>(StringComparer.Ordinal);
        readonly HashSet<string> seenRecordIds = new HashSet<string>(StringComparer.Ordinal);
        readonly HashSet<string> seenSourceHashes = new HashSet<string>(StringComparer.Ordinal);
        readonly HashSet<string> seenCaptureIds = new HashSet<string>(StringComparer.Ordinal);
        readonly HashSet<string> seenFeatureDigests = new HashSet<string>(StringComparer.Ordinal);
        long totalSourceBytes;

        public string LearnerId { get; private set; }
        public string Directory { get; private set; }
        public FullHandEnrollmentBinding Binding { get { return binding; } }
        public int RequiredRecordings { get { return binding.RequiredRecordings; } }
        /// <summary>
        /// False whenever the last scan could not read the whole directory within the accepted
        /// bound. An incomplete scan establishes NO inventory: counts are cleared, saving is
        /// refused and the gate must refuse.
        /// </summary>
        public bool InventoryComplete { get; private set; }
        /// <summary>The last refusal code. Diagnostics only; carries no path or capture data.</summary>
        public string LastFailureCode { get; private set; }
        public string LastFailureDetail { get; private set; }
        /// <summary>Records that were present but refused at reload. Never counted.</summary>
        public int RefusedRecordCount { get; private set; }

        FullHandParticipantCandidateStore(string candidateRoot, string directory, string learnerId,
                                          FullHandEnrollmentBinding binding)
        {
            this.candidateRoot = candidateRoot;
            Directory = directory;
            LearnerId = learnerId;
            this.binding = binding;
            LastFailureCode = FullHandEnrollmentCodes.None;
            LastFailureDetail = string.Empty;
        }

        /// <summary>
        /// Opens (creating only its own directory) the store for one participant. The learner
        /// id and the binding's participant id must be the SAME identity: a store is never
        /// opened for someone else's records.
        ///
        /// The existing ancestor chain is inspected for reparse points BEFORE any directory is
        /// created, so an existing junction at the candidate root or at the learner segment
        /// cannot cause a directory to be created outside the intended root; the whole chain is
        /// inspected again afterwards.
        /// </summary>
        public static FullHandParticipantCandidateStore Open(string candidateRoot, string learnerId,
                                                             FullHandEnrollmentBinding binding)
        {
            if (binding == null) throw new ArgumentNullException("binding");
            if (!IsSafeIdentitySegment(learnerId))
                throw Refuse(FullHandEnrollmentCodes.OwnerMismatch,
                    "the learner id is not a plain local identity segment");
            if (!string.Equals(learnerId, binding.ParticipantId, StringComparison.Ordinal))
                throw Refuse(FullHandEnrollmentCodes.OwnerMismatch,
                    "the store owner is not the pinned participant");
            string root = RequireSafeRoot(candidateRoot);

            string directory = Path.Combine(Path.Combine(root, learnerId), DirectoryName);

            // BEFORE creating anything: every element of the chain that already exists must be
            // a real directory and must not be a reparse point
            RequireChainSafe(directory, root);
            try
            {
                if (!System.IO.Directory.Exists(directory))
                    System.IO.Directory.CreateDirectory(directory);
            }
            catch (Exception)
            {
                throw Refuse(FullHandEnrollmentCodes.StoreUnavailable,
                    "the local record directory could not be prepared");
            }
            // and again afterwards, so a link created between the two checks is still refused
            RequireChainSafe(directory, root);

            var store = new FullHandParticipantCandidateStore(root, directory, learnerId, binding);
            store.Reload();
            return store;
        }

        // ---- counts --------------------------------------------------------------

        public int CountFor(string word)
        {
            List<FullHandParticipantCandidateRecord> records;
            return byWord.TryGetValue(word ?? string.Empty, out records) ? records.Count : 0;
        }

        public int TotalCount
        {
            get
            {
                int total = 0;
                foreach (var entry in byWord) total += entry.Value.Count;
                return total;
            }
        }

        public bool IsCollectedFor(string word)
        {
            return InventoryComplete && binding.Supports(word) &&
                   CountFor(word) >= binding.RequiredRecordings;
        }

        /// <summary>
        /// Every word of the CURRENT model vocabulary has its quorum of accepted recordings,
        /// established by a COMPLETE scan. A record-collection statement only.
        /// </summary>
        public bool AllWordsCollected
        {
            get
            {
                if (!InventoryComplete || binding.VocabularyCount == 0) return false;
                for (int index = 0; index < binding.VocabularyCount; index++)
                    if (!IsCollectedFor(binding.WordAt(index))) return false;
                return true;
            }
        }

        /// <summary>How many words of the current vocabulary are fully collected.</summary>
        public int CollectedWordCount
        {
            get
            {
                int collected = 0;
                for (int index = 0; index < binding.VocabularyCount; index++)
                    if (IsCollectedFor(binding.WordAt(index))) collected++;
                return collected;
            }
        }

        public IReadOnlyList<FullHandParticipantCandidateRecord> RecordsFor(string word)
        {
            List<FullHandParticipantCandidateRecord> records;
            if (!byWord.TryGetValue(word ?? string.Empty, out records))
                return Array.AsReadOnly(new FullHandParticipantCandidateRecord[0]);
            return records.AsReadOnly();
        }

        /// <summary>Counts for the pinned vocabulary, in vocabulary order. Never a claim.</summary>
        public string CountSummary()
        {
            var text = new StringBuilder();
            for (int index = 0; index < binding.VocabularyCount; index++)
            {
                if (index > 0) text.Append(" · ");
                string word = binding.WordAt(index);
                text.Append(word).Append(' ').Append(CountFor(word)).Append('/')
                    .Append(binding.RequiredRecordings);
            }
            if (!InventoryComplete) text.Append(" · scan=INCOMPLETE");
            return text.ToString();
        }

        // ---- reload --------------------------------------------------------------

        /// <summary>
        /// Rebuilds every count from the files on disk, re-verifying each record's integrity
        /// and identity. A stale, foreign, changed, truncated or corrupt record is refused and
        /// never counted; it is left on disk untouched, and it does not invalidate the other,
        /// independently verified records.
        ///
        /// An INCOMPLETE scan is different: it establishes no inventory at all. The counts are
        /// cleared and <see cref="InventoryComplete"/> stays false, so nothing downstream can
        /// read a partial listing as a satisfied quorum.
        /// </summary>
        public void Reload()
        {
            ClearInventory();
            LastFailureCode = FullHandEnrollmentCodes.None;
            LastFailureDetail = string.Empty;
            RefusedRecordCount = 0;

            try { RequireChainSafe(Directory, candidateRoot); }
            catch (FullHandFormatException exception)
            {
                LastFailureCode = exception.Code;
                LastFailureDetail = "the local record path was refused";
                return;
            }

            var names = new List<string>();
            try
            {
                int inspected = 0;
                foreach (string path in System.IO.Directory.EnumerateFiles(Directory))
                {
                    if (++inspected > MaxFilesInspected)
                    {
                        // the directory could not be listed completely within the bound: no
                        // inventory is established and every count stays cleared
                        LastFailureCode = FullHandEnrollmentCodes.InventoryIncomplete;
                        LastFailureDetail = "the local record directory holds more files than " +
                                            "the accepted inspection limit";
                        ClearInventory();
                        return;
                    }
                    string name = Path.GetFileName(path);
                    // every file counts towards the hard inspection bound above, but the
                    // committed binary sidecar of a healthy record is not a rejected record
                    // and must not inflate the refusal diagnostic
                    if (IsPlainRecordName(name)) names.Add(name);
                    else if (!IsPlainBinaryName(name)) RefusedRecordCount++;
                }
            }
            catch (Exception)
            {
                LastFailureCode = FullHandEnrollmentCodes.StoreUnavailable;
                LastFailureDetail = "the local record directory could not be listed";
                ClearInventory();
                return;
            }

            names.Sort(StringComparer.Ordinal);
            for (int index = 0; index < names.Count; index++)
            {
                string recordId = names[index].Substring(
                    0, names[index].Length - RecordExtension.Length);
                FullHandParticipantCandidateRecord record;
                string code, detail;
                if (!TryLoadRecord(recordId, out record, out code, out detail))
                {
                    RefusedRecordCount++;
                    LastFailureCode = code;
                    LastFailureDetail = detail;
                    continue;
                }
                if (!TryIndex(record)) RefusedRecordCount++;
            }
            InventoryComplete = true;
        }

        void ClearInventory()
        {
            byWord.Clear();
            seenRecordIds.Clear();
            seenSourceHashes.Clear();
            seenCaptureIds.Clear();
            seenFeatureDigests.Clear();
            totalSourceBytes = 0;
            InventoryComplete = false;
        }

        bool TryIndex(FullHandParticipantCandidateRecord record)
        {
            // uniqueness across ALL of record id, source hash, capture identity and digest
            if (seenRecordIds.Contains(record.RecordId) ||
                seenSourceHashes.Contains(record.SourceSha256) ||
                seenCaptureIds.Contains(record.CaptureSessionId) ||
                seenFeatureDigests.Contains(record.FeatureDigest))
                return false;

            List<FullHandParticipantCandidateRecord> records;
            if (!byWord.TryGetValue(record.Word, out records))
            {
                records = new List<FullHandParticipantCandidateRecord>();
                byWord[record.Word] = records;
            }
            if (records.Count >= MaxRecordsPerWord) return false;
            if (TotalCount >= MaxRecords) return false;
            long total;
            try { checked { total = totalSourceBytes + record.SourceByteCount; } }
            catch (OverflowException) { return false; }
            if (total > MaxTotalSourceBytes) return false;

            records.Add(record);
            seenRecordIds.Add(record.RecordId);
            seenSourceHashes.Add(record.SourceSha256);
            seenCaptureIds.Add(record.CaptureSessionId);
            seenFeatureDigests.Add(record.FeatureDigest);
            totalSourceBytes = total;
            return true;
        }

        /// <summary>
        /// Reads and re-verifies ONE record. Every required provenance field must be present
        /// exactly once, must be a plain string, and must agree with the pinned binding, with
        /// the file stem and with the recomputed capture evidence.
        /// </summary>
        bool TryLoadRecord(string recordId, out FullHandParticipantCandidateRecord record,
                           out string failureCode, out string failureDetail)
        {
            record = null;
            failureCode = FullHandEnrollmentCodes.RecordCorrupt;
            failureDetail = "a stored record was refused";

            Dictionary<string, string> fields;
            try
            {
                byte[] documentBytes = ReadBoundedFile(
                    Path.Combine(Directory, recordId + RecordExtension), MaxRecordBytes);
                string text = new UTF8Encoding(false, true).GetString(documentBytes);
                if (!TryParseFlatStringObject(text, out fields)) return false;
            }
            catch (Exception) { return false; }

            // exactly the required key set, each exactly once, and nothing else
            if (fields.Count != RequiredKeys.Length) return false;
            for (int index = 0; index < RequiredKeys.Length; index++)
                if (!fields.ContainsKey(RequiredKeys[index])) return false;

            if (!Equal(fields, "schema", Schema)) return false;
            // the explicit, exact quality declaration - never inferred
            if (!Equal(fields, "quality_status", QualityStatusModelGateAccepted)) return false;
            if (!Equal(fields, "quality_basis", QualityBasisLiteral)) return false;

            // stale owner / model / gate is a refusal, never a silently counted record
            if (!Equal(fields, "participant_id", binding.ParticipantId))
            {
                failureCode = FullHandEnrollmentCodes.OwnerMismatch;
                failureDetail = "a stored record belongs to another participant";
                return false;
            }
            if (!Equal(fields, "model_id", binding.ModelId))
            {
                failureCode = FullHandEnrollmentCodes.ModelMismatch;
                failureDetail = "a stored record was collected under another model";
                return false;
            }
            if (!Equal(fields, "gate_config_id", binding.GateConfigId) ||
                !Equal(fields, "gate_id", binding.AuthorityGateId))
            {
                failureCode = FullHandEnrollmentCodes.GateMismatch;
                failureDetail = "a stored record was collected under another gate";
                return false;
            }

            string word = fields["word"];
            if (!binding.Supports(word)) return false;

            // the ACCEPTED decision is read from the recorded evidence, and every field that
            // could make it look accepted must agree with every other one
            if (!Equal(fields, "decision", FullHandDecisions.Accepted) ||
                !Equal(fields, "method", FullHandMethods.Hybrid) ||
                !Equal(fields, "bank_id", FullHandPipelineBanks.CommonOnly) ||
                !Equal(fields, "target_supported", "true") ||
                !Equal(fields, "target_matches", "true") ||
                !Equal(fields, "declared_target", word) ||
                !Equal(fields, "top1", word))
            {
                failureCode = FullHandEnrollmentCodes.NotAccepted;
                failureDetail = "a stored record does not read as an accepted matching decision";
                return false;
            }

            string captureId = fields["capture_session_id"];
            if (!IsSafeIdentitySegment(captureId)) return false;
            // the scored sample and the capture must be the SAME single capture
            if (!Equal(fields, "query_sample_id", captureId)) return false;

            string sourceSha = NormalizeSha256(fields["source_sha256"]);
            string binarySha = NormalizeSha256(fields["binary_sha256"]);
            string digest = NormalizeSha256(fields["feature_digest"]);
            if (sourceSha == null || binarySha == null || digest == null) return false;
            if (!string.Equals(sourceSha, binarySha, StringComparison.Ordinal)) return false;

            DateTime savedUtc;
            if (!DateTime.TryParseExact(fields["saved_utc"], SavedUtcFormat,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal,
                    out savedUtc))
                return false;

            int declaredBytes = ParsePositive(fields["binary_bytes"]);
            int declaredFrames = ParsePositive(fields["frame_count"]);
            if (declaredBytes <= 0 || declaredBytes > MaxSourceBytes || declaredFrames <= 0)
                return false;

            // the file stem and the declared record id are both the DERIVED id, so a renamed,
            // copied or grafted record is refused
            string expectedId = BuildRecordId(binding, word, captureId, sourceSha, digest);
            if (!string.Equals(expectedId, recordId, StringComparison.Ordinal)) return false;
            if (!Equal(fields, "record_id", recordId)) return false;
            if (!Equal(fields, "binary_file", recordId + BinaryExtension)) return false;

            byte[] binary;
            try
            {
                binary = ReadBoundedFile(Path.Combine(Directory, recordId + BinaryExtension),
                                         declaredBytes);
            }
            catch (Exception) { return false; }
            if (binary.Length != declaredBytes) return false;
            if (!string.Equals(FullHandParticipantReferencePacket.Sha256Hex(binary), sourceSha,
                               StringComparison.Ordinal))
                return false;

            // the ONE binding between these bytes and the recorded evidence, recomputed with
            // the existing reviewed decoder and the approved geometry
            int frames;
            FullHandFeatureSequence sequence;
            try
            {
                sequence = FullHandParticipantReferencePacket
                    .DecodeRetainedCaptureFeatures(binary, out frames);
            }
            catch (Exception) { return false; }
            if (frames != declaredFrames || sequence == null || !sequence.Usable) return false;
            if (!string.Equals(FullHandFeatureDigest.Compute(sequence), digest,
                               StringComparison.Ordinal))
                return false;

            record = new FullHandParticipantCandidateRecord(recordId, word,
                binding.ParticipantId, binding.ModelId, binding.GateConfigId,
                binding.AuthorityGateId, captureId, sourceSha, digest, frames, declaredBytes,
                QualityStatusModelGateAccepted, QualityBasisLiteral, savedUtc);
            failureCode = FullHandEnrollmentCodes.None;
            failureDetail = string.Empty;
            return true;
        }

        // ---- save ----------------------------------------------------------------

        /// <summary>
        /// Commits ONE accepted recording, or refuses. The acceptance is DERIVED here from the
        /// capture evidence through the frozen evaluator; the caller-supplied decision is only
        /// cross-checked against that derivation and is never trusted on its own.
        ///
        /// The save happens BEFORE any caller increments anything: the returned count is read
        /// back from this store, so a failed or duplicate write can never move progress.
        /// </summary>
        public FullHandEnrollmentSaveResult TrySave(FullHandGameplayDecision decision,
                                                    FullHandGameCaptureResult evidence,
                                                    string promptedWord, DateTime utcNow)
        {
            string word = promptedWord ?? string.Empty;
            if (!InventoryComplete)
            {
                // FAILURE state only. A healthy store never reaches this line, so no ordinary
                // capture triggers a rescan; but a directory that has since been corrected must
                // be able to recover, so exactly ONE bounded rescan is attempted here. A still
                // incomplete scan refuses, and a partial listing is never accepted as counts.
                Reload();
                if (!InventoryComplete)
                    return Refuse(FullHandEnrollmentSaveStatus.StorageFailed, word,
                        FullHandEnrollmentCodes.InventoryIncomplete,
                        "the local record inventory is not established");
            }
            try { RequireChainSafe(Directory, candidateRoot); }
            catch (FullHandFormatException exception)
            {
                return Refuse(FullHandEnrollmentSaveStatus.StorageFailed, word, exception.Code,
                    "the local record path was refused");
            }

            if (decision == null || evidence == null)
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "this attempt carried no full-hand decision or evidence");
            if (!binding.Supports(word))
                return Refuse(FullHandEnrollmentSaveStatus.NotAccepted, word,
                    FullHandEnrollmentCodes.NotAccepted,
                    "the prompted word is outside the current model vocabulary");

            // derived from the evidence, never from decision.Kind alone
            FullHandGameplayDecision derived;
            try
            {
                derived = FullHandGameplayEvaluator.Decide(evidence, word,
                    evidence.CaptureSessionId, true);
            }
            catch (Exception)
            {
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable, "the evidence could not be re-read");
            }
            if (!IsAcceptedForBinding(derived, decision, word, binding))
                return Refuse(FullHandEnrollmentSaveStatus.NotAccepted, word,
                    FullHandEnrollmentCodes.NotAccepted,
                    "only an accepted capture of this participant, model and gate whose ranked " +
                    "winner is the prompted word is stored");

            string captureId = evidence.CaptureSessionId;
            if (!IsSafeIdentitySegment(captureId))
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "this capture carries no usable local identity");

            byte[] binary = evidence.CopySourceBinary();
            string sourceSha = NormalizeSha256(evidence.SourceSha256);
            string digest = NormalizeSha256(derived.QueryFeatureDigest);
            if (binary == null || binary.Length == 0 || sourceSha == null || digest == null)
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "this capture retained no source bytes to store");
            if (binary.Length > MaxSourceBytes)
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.BudgetExceeded,
                    "this capture is larger than the accepted record limit");
            if (!string.Equals(FullHandParticipantReferencePacket.Sha256Hex(binary), sourceSha,
                               StringComparison.Ordinal))
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "the retained bytes are not the hashed capture");

            int frames;
            FullHandFeatureSequence sequence;
            try
            {
                sequence = FullHandParticipantReferencePacket
                    .DecodeRetainedCaptureFeatures(binary, out frames);
            }
            catch (Exception)
            {
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "the retained bytes were refused by the approved decoder");
            }
            if (sequence == null || !sequence.Usable || frames <= 0)
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "the retained bytes carry no usable shape");
            if (!string.Equals(FullHandFeatureDigest.Compute(sequence), digest,
                               StringComparison.Ordinal))
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.EvidenceUnusable,
                    "the retained bytes do not reproduce the recorded feature digest");

            if (CountFor(word) >= MaxRecordsPerWord || TotalCount >= MaxRecords ||
                totalSourceBytes + binary.Length > MaxTotalSourceBytes)
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.BudgetExceeded,
                    "the local record limit for this participant is reached");

            string recordId = BuildRecordId(binding, word, captureId, sourceSha, digest);
            if (seenRecordIds.Contains(recordId) || seenSourceHashes.Contains(sourceSha) ||
                seenCaptureIds.Contains(captureId) || seenFeatureDigests.Contains(digest))
                return new FullHandEnrollmentSaveResult(FullHandEnrollmentSaveStatus.Duplicate,
                    word, CountFor(word), FullHandEnrollmentCodes.None,
                    "the same capture evidence is already stored");

            string document = BuildDocument(recordId, word, captureId, derived, sourceSha, digest,
                                            frames, binary.Length, utcNow);
            if (document == null)
                return Refuse(FullHandEnrollmentSaveStatus.Refused, word,
                    FullHandEnrollmentCodes.RecordCorrupt,
                    "the record could not be written safely");

            string code, detail;
            if (!TryCommit(recordId, document, binary, out code, out detail))
                return Refuse(FullHandEnrollmentSaveStatus.StorageFailed, word, code, detail);

            // the count is read back from the store, never assumed from the write
            FullHandParticipantCandidateRecord committed;
            if (!TryLoadRecord(recordId, out committed, out code, out detail) ||
                !TryIndex(committed))
                return Refuse(FullHandEnrollmentSaveStatus.StorageFailed, word,
                    string.IsNullOrEmpty(code) ? FullHandEnrollmentCodes.WriteFailed : code,
                    "the committed record did not verify on read-back");

            return new FullHandEnrollmentSaveResult(FullHandEnrollmentSaveStatus.Saved, word,
                CountFor(word), FullHandEnrollmentCodes.None, string.Empty);
        }

        /// <summary>
        /// The one acceptance rule, shared by the store and by the quest route: the decision
        /// DERIVED from this capture's evidence must be Accepted for this participant, model
        /// and gate, its ranked winner must be the prompted word, and the decision the caller
        /// carried must agree with it.
        /// </summary>
        public static bool IsAcceptedForBinding(FullHandGameplayDecision derived,
                                                FullHandGameplayDecision carried, string word,
                                                FullHandEnrollmentBinding binding)
        {
            if (derived == null || binding == null || string.IsNullOrEmpty(word)) return false;
            if (derived.Kind != FullHandGameplayDecisionKind.Accepted) return false;
            if (!string.Equals(derived.Top1, word, StringComparison.Ordinal) ||
                !string.Equals(derived.DeclaredTarget, word, StringComparison.Ordinal))
                return false;
            if (!string.Equals(derived.Method, FullHandMethods.Hybrid, StringComparison.Ordinal) ||
                !string.Equals(derived.BankId, FullHandPipelineBanks.CommonOnly,
                               StringComparison.Ordinal))
                return false;
            if (!string.Equals(derived.ParticipantId, binding.ParticipantId,
                               StringComparison.Ordinal) ||
                !string.Equals(derived.ModelId, binding.ModelId, StringComparison.Ordinal) ||
                !string.Equals(derived.GateConfigId, binding.GateConfigId,
                               StringComparison.Ordinal) ||
                !string.Equals(derived.GateId, binding.AuthorityGateId, StringComparison.Ordinal))
                return false;
            if (carried == null) return true;
            return carried.Kind == FullHandGameplayDecisionKind.Accepted &&
                   string.Equals(carried.CaptureSessionId, derived.CaptureSessionId,
                                 StringComparison.Ordinal) &&
                   string.Equals(carried.QueryFeatureDigest, derived.QueryFeatureDigest,
                                 StringComparison.Ordinal) &&
                   string.Equals(carried.Top1, derived.Top1, StringComparison.Ordinal);
        }

        /// <summary>
        /// Bounded, rename-based commit. The binary lands first and the record document is the
        /// ONLY commit marker, so an interruption at any point leaves either a staging file or
        /// a binary without a record - and neither is ever counted. Every path used here is
        /// composed from the validated record id and this store's own validated directory.
        /// </summary>
        bool TryCommit(string recordId, string document, byte[] binary, out string failureCode,
                       out string failureDetail)
        {
            failureCode = FullHandEnrollmentCodes.WriteFailed;
            failureDetail = "the record could not be committed";
            string binaryPath = Path.Combine(Directory, recordId + BinaryExtension);
            string recordPath = Path.Combine(Directory, recordId + RecordExtension);
            string binaryStaging = binaryPath + StagingExtension;
            string recordStaging = recordPath + StagingExtension;
            try
            {
                if (File.Exists(recordPath))
                {
                    failureCode = FullHandEnrollmentCodes.RecordCorrupt;
                    failureDetail = "a record with this identity already exists";
                    return false;
                }
                RequireNotReparseLeaf(binaryPath);
                RequireNotReparseLeaf(recordPath);
                RequireNotReparseLeaf(binaryStaging);
                RequireNotReparseLeaf(recordStaging);

                WriteAllBytesExclusive(binaryStaging, binary);
                WriteAllBytesExclusive(recordStaging, new UTF8Encoding(false).GetBytes(document));
                if (File.Exists(binaryPath)) File.Delete(binaryPath);
                File.Move(binaryStaging, binaryPath);
                File.Move(recordStaging, recordPath);
                failureCode = FullHandEnrollmentCodes.None;
                failureDetail = string.Empty;
                return true;
            }
            catch (Exception)
            {
                // only this store's own derived staging paths are ever removed
                TryDeleteOwnStaging(binaryStaging);
                TryDeleteOwnStaging(recordStaging);
                return false;
            }
        }

        static void WriteAllBytesExclusive(string path, byte[] bytes)
        {
            if (File.Exists(path)) File.Delete(path);
            using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write,
                                               FileShare.None))
            {
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush(true);
            }
        }

        void TryDeleteOwnStaging(string path)
        {
            try
            {
                if (!path.StartsWith(Directory, StringComparison.Ordinal)) return;
                if (!path.EndsWith(StagingExtension, StringComparison.Ordinal)) return;
                string name = Path.GetFileName(path);
                if (!name.StartsWith(RecordIdPrefix, StringComparison.Ordinal)) return;
                if (!File.Exists(path)) return;
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) return;
                File.Delete(path);
            }
            catch (Exception) { /* best effort, on this store's own staging file only */ }
        }

        FullHandEnrollmentSaveResult Refuse(FullHandEnrollmentSaveStatus status, string word,
                                            string code, string detail)
        {
            LastFailureCode = code;
            LastFailureDetail = detail;
            return new FullHandEnrollmentSaveResult(status, word, CountFor(word), code, detail);
        }

        // ---- the record document -------------------------------------------------

        /// <summary>
        /// Hand-built from values that were all validated as plain tokens, so no capture or
        /// word text can shape the document and no escaping question can arise.
        /// </summary>
        string BuildDocument(string recordId, string word, string captureId,
                             FullHandGameplayDecision derived, string sourceSha, string digest,
                             int frames, int byteCount, DateTime utcNow)
        {
            string savedUtc = utcNow.ToUniversalTime()
                .ToString(SavedUtcFormat, CultureInfo.InvariantCulture);
            var values = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                { "schema", Schema },
                { "record_id", recordId },
                { "participant_id", binding.ParticipantId },
                { "model_id", binding.ModelId },
                { "gate_config_id", binding.GateConfigId },
                { "gate_id", binding.AuthorityGateId },
                { "method", derived.Method },
                { "bank_id", derived.BankId },
                { "word", word },
                { "declared_target", derived.DeclaredTarget },
                { "top1", derived.Top1 },
                { "decision", FullHandDecisions.Accepted },
                { "target_supported", "true" },
                { "target_matches", "true" },
                { "capture_session_id", captureId },
                { "query_sample_id", derived.QuerySampleId },
                { "feature_digest", digest },
                { "source_sha256", sourceSha },
                { "binary_file", recordId + BinaryExtension },
                { "binary_sha256", sourceSha },
                { "binary_bytes", byteCount.ToString(CultureInfo.InvariantCulture) },
                { "frame_count", frames.ToString(CultureInfo.InvariantCulture) },
                { "quality_status", QualityStatusModelGateAccepted },
                { "quality_basis", QualityBasisLiteral },
                { "saved_utc", savedUtc },
            };
            var text = new StringBuilder("{");
            for (int index = 0; index < RequiredKeys.Length; index++)
            {
                string key = RequiredKeys[index];
                string value;
                if (!values.TryGetValue(key, out value) || !IsPlainToken(value, 256)) return null;
                if (index > 0) text.Append(',');
                text.Append('"').Append(key).Append("\":\"").Append(value).Append('"');
            }
            return text.Append('}').ToString();
        }

        /// <summary>
        /// A strict reader for the flat string document this store writes: one JSON object of
        /// plain string members. A missing brace, a duplicate key, a non-string value, an
        /// escape sequence, a nested value or any trailing content is a refusal.
        ///
        /// This is NOT a general JSON parser and makes no claim to detect every possible
        /// corruption of an arbitrary document; it accepts exactly the shape written here.
        /// </summary>
        internal static bool TryParseFlatStringObject(string text,
                                                      out Dictionary<string, string> fields)
        {
            fields = new Dictionary<string, string>(StringComparer.Ordinal);
            if (string.IsNullOrEmpty(text)) return false;
            int at = SkipWhitespace(text, 0);
            if (at >= text.Length || text[at] != '{') return false;
            at = SkipWhitespace(text, at + 1);
            if (at < text.Length && text[at] == '}')
                return SkipWhitespace(text, at + 1) >= text.Length;
            while (true)
            {
                string key;
                if (!TryReadPlainString(text, ref at, out key)) return false;
                at = SkipWhitespace(text, at);
                if (at >= text.Length || text[at] != ':') return false;
                at = SkipWhitespace(text, at + 1);
                string value;
                if (!TryReadPlainString(text, ref at, out value)) return false;
                if (fields.ContainsKey(key)) return false;      // a duplicate key is a refusal
                if (fields.Count >= 64) return false;
                fields[key] = value;
                at = SkipWhitespace(text, at);
                if (at < text.Length && text[at] == ',') { at = SkipWhitespace(text, at + 1); continue; }
                if (at < text.Length && text[at] == '}')
                    return SkipWhitespace(text, at + 1) >= text.Length;
                return false;
            }
        }

        static bool TryReadPlainString(string text, ref int at, out string value)
        {
            value = null;
            if (at >= text.Length || text[at] != '"') return false;
            int start = at + 1;
            int index = start;
            while (index < text.Length && text[index] != '"')
            {
                // an escape is never written here; refusing it keeps this reader exact
                if (text[index] == '\\' || text[index] < ' ') return false;
                index++;
            }
            if (index >= text.Length) return false;
            value = text.Substring(start, index - start);
            at = index + 1;
            return true;
        }

        static int SkipWhitespace(string text, int index)
        {
            while (index < text.Length && char.IsWhiteSpace(text[index])) index++;
            return index;
        }

        static bool Equal(Dictionary<string, string> fields, string key, string expected)
        {
            string value;
            return fields.TryGetValue(key, out value) &&
                   string.Equals(value, expected, StringComparison.Ordinal);
        }

        /// <summary>
        /// Deterministic, safe file stem. It is a hash of the pinned identities and this
        /// capture's own identities, so no unsanitized capture or word text ever reaches a
        /// path, and the same evidence always resolves to the same single record.
        /// </summary>
        public static string BuildRecordId(FullHandEnrollmentBinding binding, string word,
                                           string captureSessionId, string sourceSha256,
                                           string featureDigest)
        {
            string material = Schema + "|" + binding.ParticipantId + "|" + binding.ModelId + "|" +
                              binding.GateConfigId + "|" + binding.AuthorityGateId + "|" + word +
                              "|" + captureSessionId + "|" + sourceSha256 + "|" + featureDigest;
            string hash = FullHandParticipantReferencePacket.Sha256Hex(
                new UTF8Encoding(false).GetBytes(material));
            return RecordIdPrefix + hash.Substring(0, 40);
        }

        // ---- path guards ---------------------------------------------------------

        static string RequireSafeRoot(string candidateRoot)
        {
            bool ok;
            try
            {
                ok = !string.IsNullOrWhiteSpace(candidateRoot) &&
                     Path.IsPathFullyQualified(candidateRoot);
            }
            catch (Exception) { ok = false; }
            if (!ok)
                throw Refuse(FullHandEnrollmentCodes.RootRejected,
                    "the shared candidate root is not a fully qualified local path");
            if (candidateRoot.StartsWith("\\\\", StringComparison.Ordinal) ||
                candidateRoot.StartsWith("//", StringComparison.Ordinal) ||
                candidateRoot.IndexOf("\\\\?\\", StringComparison.Ordinal) >= 0 ||
                candidateRoot.IndexOf("\\\\.\\", StringComparison.Ordinal) >= 0 ||
                candidateRoot.IndexOf(':', 2) >= 0)
                throw Refuse(FullHandEnrollmentCodes.RootRejected,
                    "a UNC, device or stream path is not a local candidate root");
            foreach (string segment in candidateRoot.Split('/', '\\'))
                if (segment == "." || segment == "..")
                    throw Refuse(FullHandEnrollmentCodes.RootRejected,
                        "the candidate root carries a traversal segment");
            return candidateRoot.TrimEnd(Path.DirectorySeparatorChar,
                                         Path.AltDirectorySeparatorChar);
        }

        /// <summary>
        /// Every element from the store directory up to the filesystem root that ALREADY
        /// exists must be a real directory and must not be a reparse point, and the record
        /// directory must still resolve below the candidate root. Called before the directory
        /// is created, after it is created, and again on every reload and every save.
        /// </summary>
        static void RequireChainSafe(string directory, string candidateRoot)
        {
            try
            {
                string resolvedRoot = Path.GetFullPath(candidateRoot)
                    .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                string resolved = Path.GetFullPath(directory)
                    .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                if (!resolved.StartsWith(resolvedRoot + Path.DirectorySeparatorChar,
                                         StringComparison.Ordinal))
                    throw Refuse(FullHandEnrollmentCodes.RootRejected,
                        "the record directory does not resolve below the candidate root");

                var info = new DirectoryInfo(resolved);
                while (info != null)
                {
                    if (File.Exists(info.FullName))
                        throw Refuse(FullHandEnrollmentCodes.RootRejected,
                            "a path element is a file where a directory is required");
                    if (info.Exists && (info.Attributes & FileAttributes.ReparsePoint) != 0)
                        throw Refuse(FullHandEnrollmentCodes.RootRejected,
                            "the local record path crosses a reparse point");
                    info = info.Parent;
                }
            }
            catch (FullHandFormatException) { throw; }
            catch (Exception)
            {
                throw Refuse(FullHandEnrollmentCodes.RootRejected,
                    "the local record path could not be inspected");
            }
        }

        static void RequireNotReparseLeaf(string path)
        {
            if (!File.Exists(path)) return;
            if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                throw Refuse(FullHandEnrollmentCodes.RootRejected,
                    "a record leaf is a reparse point");
        }

        byte[] ReadBoundedFile(string path, long limit)
        {
            RequireNotReparseLeaf(path);
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                                               FileShare.Read))
            {
                long length = stream.Length;
                if (length <= 0 || length > limit)
                    throw new InvalidOperationException("bounded read refused");
                var buffer = new byte[(int)length];
                int at = 0;
                while (at < buffer.Length)
                {
                    int read = stream.Read(buffer, at, buffer.Length - at);
                    if (read <= 0) throw new InvalidOperationException("truncated read");
                    at += read;
                }
                if (stream.ReadByte() != -1)
                    throw new InvalidOperationException("trailing bytes");
                return buffer;
            }
        }

        static bool IsPlainRecordName(string name)
        {
            return IsPlainStemName(name, RecordExtension);
        }

        /// <summary>The committed binary sidecar of a record, not a rejected document.</summary>
        static bool IsPlainBinaryName(string name)
        {
            return IsPlainStemName(name, BinaryExtension);
        }

        static bool IsPlainStemName(string name, string extension)
        {
            if (string.IsNullOrEmpty(name) || name.Length > 128) return false;
            if (!name.EndsWith(extension, StringComparison.Ordinal)) return false;
            string stem = name.Substring(0, name.Length - extension.Length);
            if (!stem.StartsWith(RecordIdPrefix, StringComparison.Ordinal) ||
                stem.Length != RecordIdLength)
                return false;
            for (int index = RecordIdPrefix.Length; index < stem.Length; index++)
            {
                char c = stem[index];
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
            }
            return true;
        }

        public static bool IsSafeIdentitySegment(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > MaxLearnerIdLength) return false;
            if (value[0] == '.' || value.IndexOf("..", StringComparison.Ordinal) >= 0) return false;
            for (int index = 0; index < value.Length; index++)
            {
                char c = value[index];
                bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
                if (!allowed) return false;
            }
            return true;
        }

        static bool IsPlainToken(string value, int maxLength)
        {
            if (value == null || value.Length == 0 || value.Length > maxLength) return false;
            for (int index = 0; index < value.Length; index++)
            {
                char c = value[index];
                bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' ||
                               c == ':' || c == '+';
                if (!allowed) return false;
            }
            return true;
        }

        static int ParsePositive(string token)
        {
            int value;
            if (token == null || token.Length > 12 ||
                !int.TryParse(token, NumberStyles.None, CultureInfo.InvariantCulture, out value))
                return 0;
            return value;
        }

        static string NormalizeSha256(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length != 64) return null;
            var text = new StringBuilder(64);
            for (int index = 0; index < 64; index++)
            {
                char c = value[index];
                bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                           (c >= 'A' && c <= 'F');
                if (!hex) return null;
                text.Append(char.ToLowerInvariant(c));
            }
            return text.ToString();
        }

        static FullHandFormatException Refuse(string code, string detail)
        {
            return new FullHandFormatException(code, detail);
        }
    }
}
