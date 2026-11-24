// F22 - opt-in LOCAL evidence capture for the F21 game seam.
//
// This file adds NO geometry, NO classifier, NO threshold, NO number and NO model. It never
// adapts, normalizes or scores anything: it only serializes what F20 already decided and the
// EXACT capture binary that FullHandGameCaptureProcessor already produced. It is a wrapper,
// so the recognition result it returns is the very same object it was given.
//
// It DOES add work on the calling thread: local serialization, synchronous file IO, and one
// SHA-256 integrity hash of the retained binary that is deliberately recomputed here to
// verify the pair before it is written. That can take time. What it does not do is adapt,
// normalize or score the capture again, or change the result it returns.
//
// Nothing here activates by default. Without WIS_FULL_HAND_PIPELINE_TRACE pointing at a
// fully qualified output directory - and without F21 itself being explicitly enabled - no
// writer is constructed, no directory is touched and no byte is retained after scoring.
//
// Explicitly NOT here: enrollment, profile writes, raw camera video, backup, public export,
// old-participant reference import, retries, re-scoring, and any claim that a declared target
// is ground truth.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    /// <summary>
    /// Safe, non-secret trace write codes. These never carry a path, a participant identity,
    /// a declared target or any capture content, so they are safe to surface in diagnostics.
    /// </summary>
    public static class FullHandTraceCodes
    {
        public const string None = "";
        /// <summary>Nothing was requested of the writer yet.</summary>
        public const string NotAttempted = "TRACE_NOT_ATTEMPTED";
        /// <summary>This capture id was already recorded by this writer session.</summary>
        public const string DuplicateCapture = "TRACE_DUPLICATE_CAPTURE";
        /// <summary>The wrapped processor returned no result at all.</summary>
        public const string ResultAbsent = "TRACE_RESULT_ABSENT";
        /// <summary>A successful outcome arrived without the retained source binary.</summary>
        public const string BinaryUnavailable = "TRACE_BINARY_UNAVAILABLE";
        /// <summary>
        /// The retained binary does not match the hash the pipeline was handed, or that hash
        /// is absent. The pair is not written: unverifiable evidence is not evidence.
        /// </summary>
        public const string BinaryHashMismatch = "TRACE_BINARY_HASH_MISMATCH";
        /// <summary>The configured directory is not a usable fully qualified new directory.</summary>
        public const string DirectoryUnusable = "TRACE_DIRECTORY_UNUSABLE";
        /// <summary>The binary or the manifest could not be written.</summary>
        public const string WriteFailed = "TRACE_WRITE_FAILED";
    }

    /// <summary>
    /// Writes one binary/manifest pair per capture into ONE operator-configured directory.
    ///
    /// Ordering is deliberate: the binary is written FIRST, then the manifest is written to a
    /// STAGING name and only moved to its final <c>.manifest.json</c> name once that write
    /// completed. So a file with the final manifest name is a completion marker; a leftover
    /// staging file or a binary with no final manifest is incomplete evidence and must never
    /// be read as a finished record. The manifest carries the binary's byte count and
    /// SHA-256, so a corrupt pair is detectable without trusting the writer. Every file is
    /// created with <see cref="FileMode.CreateNew"/> and moved without overwrite, under a
    /// generated safe name; no participant id, capture id or declared target is ever a path.
    /// </summary>
    public sealed class FullHandCaptureTraceWriter
    {
        public const string Schema = "wis_full_hand_capture_trace_v1";
        public const string BinaryExtension = ".bin";
        public const string ManifestExtension = ".manifest.json";
        /// <summary>Never a completed record. Present only while a manifest is being written.</summary>
        public const string StagingExtension = ".manifest.partial";
        /// <summary>The declared target is a REQUESTED EXPRESSION, never ground truth.</summary>
        public const string QualityLabel = "UNASSESSED";

        readonly object gate = new object();
        readonly HashSet<string> recordedCaptureIds = new HashSet<string>(StringComparer.Ordinal);
        readonly string directory;
        int sequence;

        public string LastWriteCode { get; private set; }
        /// <summary>A bare exception type name at most. Never a path and never capture data.</summary>
        public string LastWriteDetail { get; private set; }
        /// <summary>Number of manifests this writer session completed.</summary>
        public int RecordCount { get; private set; }

        public FullHandCaptureTraceWriter(string outputDirectory)
        {
            if (outputDirectory == null) throw new ArgumentNullException("outputDirectory");
            directory = outputDirectory;
            LastWriteCode = FullHandTraceCodes.NotAttempted;
            LastWriteDetail = string.Empty;
        }

        /// <summary>
        /// Null when the operator did not configure a trace directory. This is the default
        /// path: no writer, therefore no retention and no write of any kind.
        /// </summary>
        public static FullHandCaptureTraceWriter CreateOrNull(string outputDirectory)
        {
            if (string.IsNullOrWhiteSpace(outputDirectory)) return null;
            return new FullHandCaptureTraceWriter(outputDirectory);
        }

        /// <summary>
        /// Records one capture. Returns a safe code; it NEVER throws, so a persistence fault
        /// can never reach the caller and can never alter a recognition result.
        /// </summary>
        public string Record(FullHandGameCaptureResult result)
        {
            string code;
            string detail = string.Empty;
            try { code = RecordCore(result, out detail); }
            catch (Exception exception)
            {
                code = FullHandTraceCodes.WriteFailed;
                detail = exception.GetType().Name;
            }
            lock (gate)
            {
                LastWriteCode = code;
                LastWriteDetail = detail ?? string.Empty;
            }
            return code;
        }

        string RecordCore(FullHandGameCaptureResult result, out string detail)
        {
            detail = string.Empty;
            if (result == null) return FullHandTraceCodes.ResultAbsent;

            // one capture id is recorded at most once per writer session: no hidden retry,
            // no duplicate record and therefore no second scoring pass anywhere
            string captureId = result.CaptureSessionId ?? string.Empty;
            lock (gate)
            {
                if (!recordedCaptureIds.Add(captureId))
                    return FullHandTraceCodes.DuplicateCapture;
            }

            byte[] binary = null;
            string binarySha256 = null;
            if (result.Succeeded)
            {
                binary = result.CopySourceBinary();
                if (binary == null || binary.Length == 0)
                    // a successful outcome with no retained binary is never "completed" with
                    // an invented or empty input; the pair is simply not written
                    return FullHandTraceCodes.BinaryUnavailable;

                // computed ONCE here, then reused verbatim in the manifest: the bytes we are
                // about to persist must be the bytes the pipeline was actually handed
                binarySha256 = Sha256Hex(binary);
                string declared = NormalizeSha256(result.SourceSha256);
                if (declared == null || !string.Equals(declared, binarySha256, StringComparison.Ordinal))
                    return FullHandTraceCodes.BinaryHashMismatch;
            }

            string safeDirectory;
            if (!TryPrepareDirectory(out safeDirectory, out detail))
                return FullHandTraceCodes.DirectoryUnusable;

            string baseName = NextSafeName(captureId);
            string binaryName = baseName + BinaryExtension;
            string stagingPath = Path.Combine(safeDirectory, baseName + StagingExtension);
            string manifestPath = Path.Combine(safeDirectory, baseName + ManifestExtension);

            try
            {
                if (binary != null)
                    CreateNewFile(Path.Combine(safeDirectory, binaryName), binary);

                string manifest = BuildManifest(result, binary, binarySha256,
                                                binary == null ? null : binaryName);
                // staged first: a half-written file can never wear the final manifest name
                CreateNewFile(stagingPath, new UTF8Encoding(false).GetBytes(manifest));
                // and the move is the LAST step - this overload never overwrites
                File.Move(stagingPath, manifestPath);
            }
            catch (Exception exception)
            {
                detail = exception.GetType().Name;
                // best effort only: if this fails the staging file stays behind, which is
                // exactly what it means - an incomplete record, not a completed one
                try { if (File.Exists(stagingPath)) File.Delete(stagingPath); }
                catch (Exception) { }
                return FullHandTraceCodes.WriteFailed;
            }

            lock (gate) { RecordCount++; }
            return FullHandTraceCodes.None;
        }

        static void CreateNewFile(string path, byte[] contents)
        {
            using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write,
                                               FileShare.None))
            {
                stream.Write(contents, 0, contents.Length);
                stream.Flush();
            }
        }

        bool TryPrepareDirectory(out string safeDirectory, out string detail)
        {
            safeDirectory = null;
            detail = string.Empty;
            try
            {
                if (string.IsNullOrWhiteSpace(directory)) return false;
                if (!Path.IsPathFullyQualified(directory)) return false;
                foreach (string segment in directory.Split('/', '\\'))
                    if (segment == "." || segment == "..") return false;
                Directory.CreateDirectory(directory);
                safeDirectory = directory;
                return true;
            }
            catch (Exception exception)
            {
                detail = exception.GetType().Name;
                return false;
            }
        }

        /// <summary>
        /// A generated name only: an ordinal, a digest of the capture id and a fresh random
        /// token. No participant, target or capture string ever reaches the filesystem.
        /// </summary>
        string NextSafeName(string captureId)
        {
            int ordinal;
            lock (gate) { ordinal = ++sequence; }
            string token = Sha256Hex(new UTF8Encoding(false).GetBytes(captureId ?? string.Empty));
            return "cap_" + ordinal.ToString("D6", CultureInfo.InvariantCulture) + "_" +
                   token.Substring(0, 16) + "_" +
                   Guid.NewGuid().ToString("N").Substring(0, 8);
        }

        // ---- manifest ----------------------------------------------------------

        string BuildManifest(FullHandGameCaptureResult result, byte[] binary,
                             string verifiedBinarySha256, string binaryName)
        {
            var text = new StringBuilder(4096);
            text.Append('{');
            Field(text, "schema", Schema, true);
            Field(text, "record_kind", result.Succeeded ? "capture_outcome" : "technical_refusal", false);
            Field(text, "written_utc", DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture), false);
            Field(text, "capture_session_id", result.CaptureSessionId, false);
            Raw(text, "frame_count", Integer(result.FrameCount));

            // A declared target is what the game ASKED FOR. It is never evidence that the
            // capture actually is that expression, so nothing here is labelled or enrollable.
            Field(text, "quality_label", QualityLabel, false);
            Raw(text, "eligible_for_enrollment", "false");
            Raw(text, "declared_target_is_requested_expression", "true");
            Raw(text, "declared_target_is_ground_truth", "false");

            FullHandRecognitionOutcome outcome = result.Outcome;
            if (!result.Succeeded || outcome == null)
            {
                // a technical refusal is preserved AS a refusal: no score, no binary, no
                // fabricated decision and no participant claim of any kind
                Raw(text, "scores_present", "false");
                Raw(text, "source_binary", "null");
                Raw(text, "personalization_claimed", "false");
                text.Append(",\"failure\":{");
                Field(text, "code", result.FailureCode, true);
                Field(text, "detail", result.FailureDetail, false);
                text.Append('}');
                text.Append('}');
                return text.ToString();
            }

            Raw(text, "scores_present", "true");
            Field(text, "participant_id", outcome.ParticipantId, false);
            Field(text, "declared_target", outcome.DeclaredTarget, false);
            Raw(text, "declared_target_supported", Bool(outcome.TargetSupported));
            Field(text, "query_sample_id", outcome.QuerySampleId, false);
            // the ONE digest every method entry below is bound to: same input, three methods,
            // two banks - which is exactly what makes the comparison graphs defensible
            Field(text, "query_feature_digest", outcome.QueryFeatureDigest, false);
            Field(text, "common_model_id", outcome.CommonModelId, false);
            Field(text, "gate_config_id", outcome.GateConfigId, false);
            Raw(text, "reused_common_evaluation", Bool(outcome.ReusedCommonEvaluation));
            Raw(text, "scoring_passes", Integer(outcome.ScoringPasses));

            FullHandBankOutcome after = outcome.After;
            int participantReferences = after == null ? 0 : after.ParticipantReferenceCount;
            // with zero current-participant references there is nothing personalized to claim
            Raw(text, "personalization_claimed", Bool(participantReferences > 0));

            text.Append(",\"source_binary\":{");
            Field(text, "file", binaryName, true);
            Raw(text, "byte_count", Integer(binary == null ? 0 : binary.Length));
            // the verification hash computed once in RecordCore, reused verbatim
            Field(text, "sha256", verifiedBinarySha256, false);
            // the hash the pipeline itself was handed; verified equal to sha256 before any
            // file was created, so a swapped or truncated file is detectable
            Field(text, "source_sha256_from_pipeline", result.SourceSha256, false);
            text.Append('}');

            text.Append(",\"banks\":{");
            text.Append("\"common_only\":");
            AppendBank(text, outcome.Before, outcome.QueryFeatureDigest);
            text.Append(",\"common_plus_current_participant\":");
            AppendBank(text, outcome.After, outcome.QueryFeatureDigest);
            text.Append('}');

            text.Append('}');
            return text.ToString();
        }

        static void AppendBank(StringBuilder text, FullHandBankOutcome bank, string queryDigest)
        {
            if (bank == null) { text.Append("null"); return; }
            text.Append('{');
            Field(text, "bank_id", bank.BankId, true);
            Raw(text, "effective_reference_count", Integer(bank.EffectiveReferenceCount));
            Raw(text, "common_reference_count", Integer(bank.CommonReferenceCount));
            Raw(text, "participant_reference_count", Integer(bank.ParticipantReferenceCount));
            Raw(text, "excluded_common_reference_count", Integer(bank.ExcludedCommonReferenceCount));
            Raw(text, "excluded_participant_reference_count",
                Integer(bank.ExcludedParticipantReferenceCount));
            text.Append(",\"methods\":[");
            for (int index = 0; index < FullHandMethods.All.Length; index++)
            {
                if (index > 0) text.Append(',');
                AppendMethod(text, bank.Decision(FullHandMethods.All[index]), queryDigest);
            }
            text.Append(']');
            text.Append('}');
        }

        static void AppendMethod(StringBuilder text, FullHandMethodDecision decision,
                                 string queryDigest)
        {
            if (decision == null) { text.Append("null"); return; }
            text.Append('{');
            Field(text, "method", decision.Method, true);
            Field(text, "query_feature_digest", queryDigest, false);
            Field(text, "status", decision.Status, false);
            Field(text, "decision", decision.Decision, false);
            Field(text, "reason", decision.Reason, false);
            Field(text, "gate_id", decision.GateId, false);
            Field(text, "top1", decision.Top1, false);
            Number(text, "score", decision.Score);
            Number(text, "margin", decision.Margin);
            Number(text, "quality", decision.Quality);
            Raw(text, "failed_pairs", Integer(decision.FailedPairs));
            Raw(text, "target_supported", Bool(decision.TargetSupported));
            Raw(text, "target_matches", Bool(decision.TargetMatches));

            // NOT sorted: these are the raw per-class scores in the bank's own class order,
            // each kept with its own class name. Nothing here re-scores or re-ranks; "top1"
            // above is F20's own selection.
            string[] classes = decision.CopyClasses();
            double[] raw = decision.CopyRawScores();
            text.Append(",\"class_scores\":[");
            for (int index = 0; index < classes.Length; index++)
            {
                if (index > 0) text.Append(',');
                text.Append('{');
                Field(text, "class", classes[index], true);
                Number(text, "score", index < raw.Length ? raw[index] : double.NaN);
                text.Append('}');
            }
            text.Append(']');

            string[] missing = decision.CopyClassesWithoutEvidence();
            text.Append(",\"classes_without_evidence\":[");
            for (int index = 0; index < missing.Length; index++)
            {
                if (index > 0) text.Append(',');
                Quote(text, missing[index]);
            }
            text.Append(']');
            text.Append('}');
        }

        // ---- json primitives ---------------------------------------------------

        /// <summary>
        /// A non-finite value is reported as JSON null with an explicit "_finite": false. It
        /// is never coerced to 0.0, which would read as a real score, and it never becomes a
        /// successful rejection.
        /// </summary>
        static void Number(StringBuilder text, string name, double value)
        {
            bool finite = !double.IsNaN(value) && !double.IsInfinity(value);
            text.Append(",\"").Append(name).Append("\":");
            text.Append(finite ? value.ToString("R", CultureInfo.InvariantCulture) : "null");
            text.Append(",\"").Append(name).Append("_finite\":").Append(finite ? "true" : "false");
        }

        static void Field(StringBuilder text, string name, string value, bool first)
        {
            if (!first) text.Append(',');
            text.Append('"').Append(name).Append("\":");
            Quote(text, value);
        }

        static void Raw(StringBuilder text, string name, string literal)
        {
            text.Append(",\"").Append(name).Append("\":").Append(literal);
        }

        static string Integer(int value)
        {
            return value.ToString(CultureInfo.InvariantCulture);
        }

        static string Bool(bool value) { return value ? "true" : "false"; }

        /// <summary>DEL: the one control character that is not below U+0020.</summary>
        const char Delete = (char)0x7F;

        /// <summary>Null is a JSON null: an absent string is never invented as "".</summary>
        static void Quote(StringBuilder text, string value)
        {
            if (value == null) { text.Append("null"); return; }
            text.Append('"');
            for (int index = 0; index < value.Length; index++)
            {
                char c = value[index];
                if (c == '"') text.Append("\\\"");
                else if (c == '\\') text.Append("\\\\");
                else if (c == '\n') text.Append("\\n");
                else if (c == '\r') text.Append("\\r");
                else if (c == '\t') text.Append("\\t");
                else if (c < ' ' || c == Delete)
                    text.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                else text.Append(c);
            }
            text.Append('"');
        }

        /// <summary>Null when absent or not the 64-hex shape. Never invents a hash.</summary>
        public static string NormalizeSha256(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length != 64) return null;
            var text = new StringBuilder(64);
            for (int index = 0; index < 64; index++)
            {
                char c = value[index];
                bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
                if (!hex) return null;
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
    }

    /// <summary>
    /// The trace wrapper. It delegates to the reviewed processor and returns that processor's
    /// result OBJECT UNCHANGED - the same reference, never a rebuilt or edited copy - so a
    /// trace fault of any kind cannot change or discard a recognition result, and the only
    /// observable effect of a failure is <see cref="LastWriteCode"/>.
    ///
    /// It DOES delay the return: recording runs synchronously on the calling thread and does
    /// hashing and file IO before Process returns. That is a latency cost the operator opts
    /// into by configuring the trace directory, not a change to any decision.
    /// </summary>
    public sealed class FullHandGameCaptureTracingProcessor : IFullHandGameCaptureProcessor
    {
        readonly IFullHandGameCaptureProcessor inner;
        readonly FullHandCaptureTraceWriter writer;

        public FullHandGameCaptureTracingProcessor(IFullHandGameCaptureProcessor inner,
                                                   FullHandCaptureTraceWriter writer)
        {
            if (inner == null) throw new ArgumentNullException("inner");
            if (writer == null) throw new ArgumentNullException("writer");
            this.inner = inner;
            this.writer = writer;
        }

        /// <summary>Diagnostics only. Safe code, no path and no capture content.</summary>
        public string LastWriteCode
        {
            get { return writer.LastWriteCode; }
        }

        public string LastWriteDetail
        {
            get { return writer.LastWriteDetail; }
        }

        public int RecordCount
        {
            get { return writer.RecordCount; }
        }

        public FullHandGameCaptureResult Process(string captureSessionId, string declaredTarget,
                                                 IReadOnlyList<RawMediaPipeObservation> observations)
        {
            FullHandGameCaptureResult result =
                inner.Process(captureSessionId, declaredTarget, observations);
            // never throws; a safe code is the only effect on the caller, but this call does
            // hash and write synchronously before the result is handed back
            writer.Record(result);
            return result;
        }
    }
}
