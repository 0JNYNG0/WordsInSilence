// F5 - opt-in, read-only full-hand observation of an already finalized capture.
//
// OFF by default. It runs only when WIS_FULL_HAND_OBSERVE=1 and both explicit absolute
// paths are supplied. It owns no camera, opens no network, touches no repository, profile,
// enrollment, quest, PlayerPrefs, Resources, StreamingAssets or scene, and never copies a
// model into a build. The bundle comes only from F4's explicit-path loader.
//
// It never changes recognition. declaredTarget is AUDIT METADATA ONLY: it is never passed
// to BuildFeatures or the scorer, and no branch anywhere depends on a word name. There is
// no ACCEPT, no threshold, no proficiency and no enrollment authority here.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    public interface IFullHandObservationSink
    {
        void WriteBytes(string name, byte[] payload);
        void WriteText(string name, string text);
    }

    /// <summary>Writes into one explicit existing absolute directory, never overwriting.</summary>
    public sealed class FullHandFileObservationSink : IFullHandObservationSink
    {
        readonly string root;

        public FullHandFileObservationSink(string root)
        {
            if (string.IsNullOrEmpty(root))
                throw new FullHandFormatException("OBSERVATION_ROOT_MISSING", "no root");
            // checked on the RAW value: GetFullPath would silently complete a relative,
            // drive-relative or root-relative path against the current directory
            if (!IsFullyQualified(root))
                throw new FullHandFormatException("OBSERVATION_ROOT_NOT_ABSOLUTE",
                    "a fully qualified directory is required");
            string full = Path.GetFullPath(root);
            if (!Directory.Exists(full))
                throw new FullHandFormatException("OBSERVATION_ROOT_MISSING",
                    "the output root must already exist");
            foreach (string segment in full.Split(Path.DirectorySeparatorChar,
                                                  Path.AltDirectorySeparatorChar))
                if (string.Equals(segment, "Assets", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(segment, "StreamingAssets", StringComparison.OrdinalIgnoreCase))
                    throw new FullHandFormatException("OBSERVATION_ROOT_REFUSED",
                        "an output root under Assets or StreamingAssets is refused");
            this.root = full;
        }

        /// <summary>
        /// True only for a path that already names its own root: a Windows drive with a
        /// separator, a UNC prefix, or a POSIX leading slash. "relative", "C:relative" and
        /// "\\relative" are all refused, because each would otherwise be completed against
        /// the current directory or the current drive.
        /// </summary>
        public static bool IsFullyQualified(string value)
        {
            if (string.IsNullOrEmpty(value)) return false;
            if (value.IndexOf("://", StringComparison.Ordinal) >= 0) return false;
            bool driveQualified = value.Length >= 3
                && ((value[0] >= 'a' && value[0] <= 'z') || (value[0] >= 'A' && value[0] <= 'Z'))
                && value[1] == ':' && (value[2] == '\\' || value[2] == '/');
            bool uncQualified = value.Length >= 2
                && (value[0] == '\\' || value[0] == '/') && (value[1] == '\\' || value[1] == '/');
            bool posixQualified = Path.DirectorySeparatorChar == '/' && value[0] == '/';
            return driveQualified || uncQualified || posixQualified;
        }

        string Resolve(string name)
        {
            // a plain file name only: a ':' would let a drive-relative name escape the root
            if (string.IsNullOrEmpty(name) || name.IndexOf('/') >= 0 || name.IndexOf('\\') >= 0
                || name.IndexOf(':') >= 0 || name == "." || name == "..")
                throw new FullHandFormatException("OBSERVATION_NAME_INVALID", "plain names only");
            return Path.Combine(root, name);
        }

        public void WriteBytes(string name, byte[] payload)
        {
            using (var stream = new FileStream(Resolve(name), FileMode.CreateNew,
                                               FileAccess.Write))
                stream.Write(payload, 0, payload.Length);
        }

        public void WriteText(string name, string text)
        {
            WriteBytes(name, new UTF8Encoding(false).GetBytes(text));
        }
    }

    /// <summary>
    /// The pure, testable seam: an explicitly supplied bundle and sink, no environment and
    /// no static state. One capture in, one private record out.
    /// </summary>
    public sealed class FullHandObservationSession
    {
        public const string RecordSchema = "wis_full_hand_observation_v1";
        public const int BinaryFormatVersion = 1;
        static readonly byte[] BinaryMagic = Encoding.ASCII.GetBytes("WISFHOB1");
        readonly FullHandRuntimeBundle bundle;
        readonly IFullHandObservationSink sink;
        readonly Action<string> log;
        // F12: optional, opt-in, review-only archive. null means the archive is off and
        // nothing beyond the existing two records is ever produced.
        readonly IFullHandReviewCandidateSink archiveSink;
        // kept for this session instance's whole lifetime. Evicting an id would let an early
        // capture be observed a second time, which the exactly-once contract forbids.
        readonly HashSet<string> observed = new HashSet<string>(StringComparer.Ordinal);

        /// <summary>Admitted attempts, and therefore the anonymous record number.</summary>
        public int ObservationCount { get; private set; }
        /// <summary>Attempts that produced an evaluated record.</summary>
        public int ScoredCount { get; private set; }
        /// <summary>Attempts that produced a failure record instead of an evaluation.</summary>
        public int FailedCount { get; private set; }
        public int SkippedDuplicateCount { get; private set; }
        public string LastStatusLine { get; private set; }

        /// <summary>Review-only archive copies actually saved. Never a promotion count.</summary>
        public int ArchiveSavedCount { get; private set; }
        /// <summary>Archive attempts refused or failed. Never affects the counters above.</summary>
        public int ArchiveFailedCount { get; private set; }
        /// <summary>Typed archive outcome only; empty while the archive is off.</summary>
        public string LastArchiveStatus { get; private set; }

        public FullHandObservationSession(FullHandRuntimeBundle bundle,
                                          IFullHandObservationSink sink,
                                          Action<string> log)
            : this(bundle, sink, log, null)
        {
        }

        /// <summary>
        /// Same session, plus an OPTIONAL review-only archive sink. A null archiveSink is
        /// exactly the old behaviour: no archive record is built and no root is touched.
        /// </summary>
        public FullHandObservationSession(FullHandRuntimeBundle bundle,
                                          IFullHandObservationSink sink,
                                          Action<string> log,
                                          IFullHandReviewCandidateSink archiveSink)
        {
            if (bundle == null) throw new ArgumentNullException("bundle");
            if (sink == null) throw new ArgumentNullException("sink");
            this.bundle = bundle;
            this.sink = sink;
            this.log = log;
            this.archiveSink = archiveSink;
            LastArchiveStatus = string.Empty;
        }

        /// <summary>At most once per capture session id. Never throws to the caller.</summary>
        public bool TryObserve(string captureSessionId, string declaredTarget,
                               IReadOnlyList<RawMediaPipeObservation> observations,
                               out string status)
        {
            status = string.Empty;
            int index = 0;
            try
            {
                if (string.IsNullOrEmpty(captureSessionId))
                { status = "NO_SESSION_ID"; Report(0, status, 0.0); return false; }
                if (!Admit(captureSessionId))
                { SkippedDuplicateCount++; status = "ALREADY_OBSERVED"; return false; }
                index = ++ObservationCount;
                var watch = System.Diagnostics.Stopwatch.StartNew();
                status = Observe(index, captureSessionId, declaredTarget, observations, watch);
                watch.Stop();
                if (status == "OBSERVED") ScoredCount++; else FailedCount++;
                Report(index, status, watch.Elapsed.TotalMilliseconds);
                return status == "OBSERVED";
            }
            catch (Exception error)
            {
                FailedCount++;
                status = "OBSERVER_ERROR_" + error.GetType().Name;
                // best effort, exception TYPE only. It uses its own name, so any record
                // already written for this number is preserved and never retried over.
                try
                {
                    if (index > 0)
                        sink.WriteText("observation_" + index.ToString("0000") + "_error.json",
                                       FailureJson(captureSessionId, declaredTarget, status,
                                                   error.GetType().Name, -1, -1, 0.0));
                }
                catch (Exception) { }
                try { Report(index, status, 0.0); } catch (Exception) { }
                return false;
            }
        }

        bool Admit(string captureSessionId)
        {
            // no eviction: a remembered id must never become admissible again
            return observed.Add(captureSessionId);
        }

        string Observe(int index, string captureSessionId, string declaredTarget,
                       IReadOnlyList<RawMediaPipeObservation> observations,
                       System.Diagnostics.Stopwatch watch)
        {
            if (observations == null || observations.Count == 0)
                return Failure(captureSessionId, declaredTarget, "EMPTY_CAPTURE",
                               "the capture carried no observation", 0, -1, index, watch);
            string adaptFailure;
            FullHandCommonCapture capture =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(observations,
                                                                       out adaptFailure);
            if (capture == null)
                return Failure(captureSessionId, declaredTarget, "ADAPT_REFUSED",
                               adaptFailure, observations.Count, -1, index, watch);
            if (capture.FrameCount != observations.Count)
                return Failure(captureSessionId, declaredTarget, "FRAME_COUNT_DISAGREEMENT",
                               "the adapter returned " + capture.FrameCount +
                               " frames for " + observations.Count + " observations",
                               observations.Count, capture.FrameCount, index, watch);

            // declaredTarget is NOT passed here, and no word name is consulted anywhere
            FullHandFeatureSequence sequence = FullHandPairGeometry.BuildFeatures(capture);
            FullHandEvaluation evaluation = FullHandScorer.Evaluate(bundle, sequence);

            var timestamps = new double[observations.Count];
            for (int frame = 0; frame < observations.Count; frame++)
                timestamps[frame] = observations[frame].TimestampSeconds;

            byte[] binary = BuildBinary(capture, timestamps);
            string binaryName = "observation_" + index.ToString("0000") + ".bin";
            string jsonName = "observation_" + index.ToString("0000") + ".json";
            sink.WriteBytes(binaryName, binary);
            sink.WriteText(jsonName, BuildJson(captureSessionId, declaredTarget, capture,
                                               sequence, evaluation, timestamps, binaryName,
                                               binary, watch.Elapsed.TotalMilliseconds));
            // both existing records are on disk and unchanged before this point
            TryArchive(index, declaredTarget, capture, timestamps, evaluation);
            return "OBSERVED";
        }

        /// <summary>
        /// Best effort, opt-in only. It reuses the SAME capture, the SAME timestamps and the
        /// SAME evaluation the record above already carries: no camera, no renormalization,
        /// no second classifier and no target ever reaches the scorer. Every fault is
        /// contained here, so the observation status and every existing counter are already
        /// final and cannot be changed by anything below.
        /// </summary>
        void TryArchive(int index, string declaredTarget, FullHandCommonCapture capture,
                        double[] timestamps, FullHandEvaluation evaluation)
        {
            if (archiveSink == null) return;         // off: no record, no root, no write
            string status;
            try
            {
                WordsInSilence.Recognition.Personalization.FullHandReviewCandidate candidate =
                    FullHandReviewArchiveBuilder.Build(
                        FullHandReviewArchiveBuilder.SampleId(index), declaredTarget,
                        capture, timestamps, evaluation, bundle.PayloadSha256);
                archiveSink.Write(candidate);
                ArchiveSavedCount++;
                status = "ARCHIVED";
            }
            catch (Exception error)
            {
                // TYPE only: no path, no message, no fabricated label of any kind
                ArchiveFailedCount++;
                status = "ARCHIVE_FAILED_" + error.GetType().Name;
            }
            LastArchiveStatus = status;
            try
            {
                if (log != null)
                    log("[FullHandArchive] capture=" + index.ToString("0000")
                        + " status=" + status
                        + " note=review_only.not_enrollment.no_promotion");
            }
            catch (Exception) { /* a failing log must never reach the capture path */ }
        }

        /// <summary>
        /// Writes the minimal private failure record for a refused input, under the same
        /// anonymous capture number. It claims no score and no binary.
        /// </summary>
        string Failure(string captureSessionId, string declaredTarget, string status,
                       string reason, int sourceCount, int frameCount, int index,
                       System.Diagnostics.Stopwatch watch)
        {
            sink.WriteText("observation_" + index.ToString("0000") + ".json",
                           FailureJson(captureSessionId, declaredTarget, status, reason,
                                       sourceCount, frameCount,
                                       watch == null ? 0.0 : watch.Elapsed.TotalMilliseconds));
            return status;
        }

        string FailureJson(string captureSessionId, string declaredTarget, string status,
                           string reason, int sourceCount, int frameCount, double milliseconds)
        {
            var text = new StringBuilder(1024);
            text.Append("{\n \"schema\": \"").Append(RecordSchema).Append("\",\n");
            text.Append(" \"record_kind\": \"failure\",\n");
            text.Append(" \"handling\": \"LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED\",\n");
            AppendBundleIdentity(text);
            text.Append(" \"private_capture_session_id\": ")
                .Append(Quote(captureSessionId ?? string.Empty)).Append(",\n");
            text.Append(" \"private_declared_target\": ")
                .Append(Quote(declaredTarget ?? string.Empty)).Append(",\n");
            text.Append(" \"declared_target_use\": \"audit metadata only; never supplied to ")
                .Append("BuildFeatures or the scorer\",\n");
            text.Append(" \"status\": ").Append(Quote(status ?? string.Empty)).Append(",\n");
            text.Append(" \"failure_reason\": ")
                .Append(string.IsNullOrEmpty(reason) ? "null" : Quote(reason)).Append(",\n");
            text.Append(" \"source_observation_count\": ").Append(sourceCount).Append(",\n");
            text.Append(" \"frame_count\": ").Append(frameCount).Append(",\n");
            text.Append(" \"counts_note\": \"-1 means the value was not available at the ")
                .Append("point of failure\",\n");
            text.Append(" \"scored\": false,\n");
            text.Append(" \"binary_file\": null,\n");
            text.Append(" \"elapsed_milliseconds\": ").Append(Number(milliseconds)).Append(",\n");
            text.Append(" \"note\": \"no evaluation was produced; this input is counted as ")
                .Append("failed and never as scored\"\n}\n");
            return text.ToString();
        }

        /// <summary>Identity of the bundle actually loaded, read from the cached instance.</summary>
        void AppendBundleIdentity(StringBuilder text)
        {
            text.Append(" \"model\": {\"payload_sha256\": ").Append(Quote(bundle.PayloadSha256))
                .Append(", \"synthetic\": ").Append(bundle.Synthetic ? "true" : "false")
                .Append(", \"exporter_version\": ")
                .Append(Quote(bundle.ExporterVersion ?? string.Empty))
                .Append(", \"reference_count\": ").Append(bundle.ReferenceCount)
                .Append(", \"vocabulary_count\": ").Append(bundle.VocabularyCount)
                .Append(", \"feature_schema\": {\"pair_schema\": ")
                .Append(Quote(FullHandPairConstants.PairSchemaId))
                .Append(", \"source_schema\": ")
                .Append(Quote(FullHandPairConstants.SourceSchemaId))
                .Append(", \"feature_point_count\": ")
                .Append(FullHandPairConstants.FeaturePointCount)
                .Append(", \"resample_frames\": ").Append(FullHandPairConstants.ResampleFrames)
                .Append(", \"fixed_feature_count\": ")
                .Append(FullHandPairConstants.FixedFeatureCount)
                .Append(", \"aligned_feature_count\": ")
                .Append(FullHandPairConstants.AlignedFeatureCount)
                .Append("}},\n");
        }

        void Report(int index, string status, double milliseconds)
        {
            // one concise line: no identifier, no path, no array, no per-frame spam
            string line = "[FullHandObserve] capture=" + index.ToString("0000")
                        + " status=" + status
                        + " ms=" + milliseconds.ToString("0.###", CultureInfo.InvariantCulture)
                        + " note=read_only_diagnostic.no_promotion";
            LastStatusLine = line;
            try { if (log != null) log(line); }
            catch (Exception) { /* a failing log must never reach the capture path */ }
        }

        // ---- binary record ----------------------------------------------------

        /// <summary>
        /// Little-endian: magic, version, dims, timestamps, [T,46,3] doubles with masked
        /// NaN retained, then the final, original-valid and interpolated masks.
        /// </summary>
        public static byte[] BuildBinary(FullHandCommonCapture capture, double[] timestamps)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            double[] coordinates = capture.CopyCoordinates();
            bool[] finalMask = capture.CopyValidMask();
            bool[] originalMask = capture.CopyOriginalValidMask();
            bool[] interpolated = capture.CopyInterpolatedMask();
            using (var stream = new MemoryStream())
            {
                stream.Write(BinaryMagic, 0, BinaryMagic.Length);
                WriteInt(stream, BinaryFormatVersion);
                WriteInt(stream, capture.FrameCount);
                WriteInt(stream, points);
                WriteInt(stream, dims);
                for (int frame = 0; frame < capture.FrameCount; frame++)
                    WriteDouble(stream, timestamps[frame]);
                for (int slot = 0; slot < coordinates.Length; slot++)
                    WriteDouble(stream, coordinates[slot]);
                WriteMask(stream, finalMask);
                WriteMask(stream, originalMask);
                WriteMask(stream, interpolated);
                return stream.ToArray();
            }
        }

        static void WriteInt(Stream stream, int value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
            stream.Write(bytes, 0, 4);
        }

        static void WriteDouble(Stream stream, double value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
            stream.Write(bytes, 0, 8);
        }

        static void WriteMask(Stream stream, bool[] mask)
        {
            for (int slot = 0; slot < mask.Length; slot++)
                stream.WriteByte(mask[slot] ? (byte)1 : (byte)0);
        }

        // ---- json record ------------------------------------------------------

        string BuildJson(string captureSessionId, string declaredTarget,
                         FullHandCommonCapture capture, FullHandFeatureSequence sequence,
                         FullHandEvaluation evaluation, double[] timestamps,
                         string binaryName, byte[] binary, double milliseconds)
        {
            int points = FullHandCommonSchema.PointCount;
            bool[] finalMask = capture.CopyValidMask();
            bool[] originalMask = capture.CopyOriginalValidMask();
            bool[] interpolated = capture.CopyInterpolatedMask();
            var text = new StringBuilder(4096);
            text.Append("{\n \"schema\": \"").Append(RecordSchema).Append("\",\n");
            text.Append(" \"record_kind\": \"observation\",\n");
            text.Append(" \"handling\": \"LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED\",\n");
            AppendBundleIdentity(text);
            text.Append(" \"binary_format_version\": ").Append(BinaryFormatVersion).Append(",\n");
            text.Append(" \"binary_file\": ").Append(Quote(binaryName)).Append(",\n");
            text.Append(" \"binary_bytes\": ").Append(binary.Length).Append(",\n");
            text.Append(" \"binary_sha256\": ").Append(Quote(Sha256Hex(binary))).Append(",\n");
            text.Append(" \"private_capture_session_id\": ").Append(Quote(captureSessionId)).Append(",\n");
            text.Append(" \"private_declared_target\": ").Append(Quote(declaredTarget ?? string.Empty)).Append(",\n");
            text.Append(" \"declared_target_use\": \"audit metadata only; never supplied to ")
                .Append("BuildFeatures or the scorer, and no branch depends on it\",\n");
            text.Append(" \"normalization\": {\"point_schema\": ")
                .Append(Quote(capture.SchemaId)).Append(", \"domain_contract\": ")
                .Append(Quote(capture.DomainContract)).Append(", \"convention_id\": ")
                .Append(Quote(capture.ConventionId)).Append(", \"image_width\": ")
                .Append(capture.ImageWidth).Append(", \"image_height\": ")
                .Append(capture.ImageHeight).Append("},\n");
            text.Append(" \"frame_count\": ").Append(capture.FrameCount).Append(",\n");
            text.Append(" \"source_timestamp_span\": {\"first\": ")
                .Append(Number(timestamps.Length > 0 ? timestamps[0] : double.NaN))
                .Append(", \"last\": ")
                .Append(Number(timestamps.Length > 0 ? timestamps[timestamps.Length - 1] : double.NaN))
                .Append(", \"duration\": ")
                .Append(Number(timestamps.Length > 1
                               ? timestamps[timestamps.Length - 1] - timestamps[0] : double.NaN))
                .Append("},\n");
            text.Append(" \"validity_totals\": {\"final_valid\": ").Append(CountTrue(finalMask))
                .Append(", \"original_valid\": ").Append(CountTrue(originalMask))
                .Append(", \"interpolated\": ").Append(CountTrue(interpolated))
                .Append(", \"slots\": ").Append(finalMask.Length).Append("},\n");
            text.Append(" \"point_order\": [");
            FullHandPointSpec[] specs = FullHandCommonSchema.CopyPoints();
            for (int index = 0; index < specs.Length; index++)
            {
                if (index > 0) text.Append(", ");
                text.Append(Quote(specs[index].Component + "." + specs[index].Name));
            }
            text.Append("],\n");
            text.Append(" \"per_joint_final_valid_frames\": {");
            AppendJointCounts(text, "pose", 0, 4, finalMask, capture.FrameCount, points);
            text.Append(", ");
            AppendJointCounts(text, "left_hand", 4, 25, finalMask, capture.FrameCount, points);
            text.Append(", ");
            AppendJointCounts(text, "right_hand", 25, 46, finalMask, capture.FrameCount, points);
            text.Append("},\n");
            text.Append(" \"hand_scales\": {\"left\": ").Append(Number(sequence.LeftHandScale))
                .Append(", \"right\": ").Append(Number(sequence.RightHandScale))
                .Append(", \"note\": \"per-capture geometry, not a personal profile\"},\n");
            text.Append(" \"sequence\": {\"usable\": ")
                .Append(sequence.Usable ? "true" : "false").Append(", \"failure\": ")
                .Append(sequence.Failure == null ? "null" : Quote(sequence.Failure))
                .Append(", \"source_frames\": ").Append(sequence.SourceFrames).Append("},\n");
            AppendSideDistances(text, capture);
            AppendMethods(text, evaluation);
            text.Append(" \"elapsed_milliseconds\": ").Append(Number(milliseconds)).Append(",\n");
            text.Append(" \"limits\": [\"read-only observation of an already finalized ")
                .Append("capture; recognition, the quest result and enrollment are untouched\", ")
                .Append("\"no ACCEPT, no threshold, no proficiency and no promotion\", ")
                .Append("\"non-finite values are explicit round-trip strings, never a fake zero\"]\n");
            text.Append("}\n");
            return text.ToString();
        }

        static void AppendJointCounts(StringBuilder text, string name, int low, int high,
                                      bool[] mask, int frames, int points)
        {
            text.Append(Quote(name)).Append(": [");
            for (int point = low; point < high; point++)
            {
                if (point > low) text.Append(", ");
                int total = 0;
                for (int frame = 0; frame < frames; frame++)
                    if (mask[frame * points + point]) total++;
                text.Append(total);
            }
            text.Append("]");
        }

        /// <summary>
        /// Same-side and cross-side mean distances between each hand's wrist point and each
        /// body wrist, over frames where BOTH are finite.
        ///
        /// This is diagnostic evidence, NOT a hand-swap classifier. Crossing the arms and
        /// timing differences confound it, and "valid" here means a finite observation, not
        /// per-joint tracking accuracy. No provider or normalization is modified.
        /// </summary>
        static void AppendSideDistances(StringBuilder text, FullHandCommonCapture capture)
        {
            text.Append(" \"wrist_side_distances\": {\n")
                .Append("  \"note\": \"diagnostic only, not a swap classifier; crossing and ")
                .Append("timing confound it, and valid means a finite observation rather ")
                .Append("than per-joint tracking accuracy\",\n");
            AppendPairDistance(text, capture, "same_side_left", 4, 2);
            text.Append(",\n");
            AppendPairDistance(text, capture, "same_side_right", 25, 3);
            text.Append(",\n");
            AppendPairDistance(text, capture, "cross_left_hand_to_right_body_wrist", 4, 3);
            text.Append(",\n");
            AppendPairDistance(text, capture, "cross_right_hand_to_left_body_wrist", 25, 2);
            text.Append("\n },\n");
        }

        static void AppendPairDistance(StringBuilder text, FullHandCommonCapture capture,
                                       string name, int handPoint, int bodyPoint)
        {
            double total = 0.0;
            int paired = 0;
            for (int frame = 0; frame < capture.FrameCount; frame++)
            {
                if (!capture.IsValid(frame, handPoint) || !capture.IsValid(frame, bodyPoint))
                    continue;
                double dx = capture.Coordinate(frame, handPoint, 0)
                          - capture.Coordinate(frame, bodyPoint, 0);
                double dy = capture.Coordinate(frame, handPoint, 1)
                          - capture.Coordinate(frame, bodyPoint, 1);
                double value = Math.Sqrt(dx * dx + dy * dy);
                if (double.IsNaN(value) || double.IsInfinity(value)) continue;
                total += value;
                paired++;
            }
            text.Append("  ").Append(Quote(name)).Append(": {\"mean_distance\": ")
                .Append(Number(paired > 0 ? total / paired : double.NaN))
                .Append(", \"paired_valid_frames\": ").Append(paired).Append("}");
        }

        static void AppendMethods(StringBuilder text, FullHandEvaluation evaluation)
        {
            text.Append(" \"query_usable\": ").Append(evaluation.QueryUsable ? "true" : "false")
                .Append(",\n \"reference_count\": ").Append(evaluation.ReferenceCount)
                .Append(",\n \"vocabulary\": [");
            for (int index = 0; index < evaluation.Vocabulary.Length; index++)
            {
                if (index > 0) text.Append(", ");
                text.Append(Quote(evaluation.Vocabulary[index]));
            }
            text.Append("],\n \"methods\": {\n");
            for (int index = 0; index < evaluation.Methods.Length; index++)
            {
                FullHandMethodResult result = evaluation.Methods[index];
                if (index > 0) text.Append(",\n");
                text.Append("  ").Append(Quote(result.Method)).Append(": {\"status\": ")
                    .Append(Quote(result.Status)).Append(", \"reason\": ")
                    .Append(result.Reason == null ? "null" : Quote(result.Reason))
                    .Append(", \"top1\": ")
                    .Append(result.Top1 == null ? "null" : Quote(result.Top1))
                    .Append(", \"score\": ").Append(Number(result.Score))
                    .Append(", \"margin\": ").Append(Number(result.Margin))
                    .Append(", \"failed_pairs\": ").Append(result.FailedPairs)
                    .Append(", \"classes_without_evidence\": [");
                string[] blocked = result.ClassesWithoutEvidence ?? new string[0];
                for (int at = 0; at < blocked.Length; at++)
                {
                    if (at > 0) text.Append(", ");
                    text.Append(Quote(blocked[at]));
                }
                text.Append("], \"raw_class_scores\": {");
                for (int at = 0; at < evaluation.Vocabulary.Length; at++)
                {
                    if (at > 0) text.Append(", ");
                    text.Append(Quote(evaluation.Vocabulary[at])).Append(": ")
                        .Append(Number(result.RawScores[at]));
                }
                text.Append("}}");
            }
            text.Append("\n },\n");
        }

        /// <summary>A non-finite value stays an explicit round-trip string, never a zero.</summary>
        static string Number(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
                return Quote(value.ToString("R", CultureInfo.InvariantCulture));
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        static int CountTrue(bool[] mask)
        {
            int total = 0;
            for (int index = 0; index < mask.Length; index++) if (mask[index]) total++;
            return total;
        }

        static string Quote(string value)
        {
            var text = new StringBuilder(value.Length + 2);
            text.Append('"');
            foreach (char character in value)
            {
                if (character == '"' || character == '\\') text.Append('\\').Append(character);
                else if (character < ' ') text.Append(' ');
                else text.Append(character);
            }
            return text.Append('"').ToString();
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

    public enum FullHandObserverState
    {
        NotAttempted,
        Disabled,
        Ready,
        Faulted,
    }

    /// <summary>
    /// The static seam the controller calls. Resolves configuration exactly once, caches the
    /// bundle, and swallows every fault so the caller is never affected.
    /// </summary>
    public static class FullHandAttemptObserver
    {
        public const string EnableVariable = "WIS_FULL_HAND_OBSERVE";
        public const string EnableValue = "1";
        public const string ManifestVariable = "WIS_FULL_HAND_BUNDLE_MANIFEST";
        public const string OutputVariable = "WIS_FULL_HAND_OBSERVATION_OUTPUT";
        /// <summary>F12: opt-in review-only archive, read ONLY once the observer is enabled.</summary>
        public const string ArchiveEnableVariable = "WIS_FULL_HAND_REVIEW_ARCHIVE";
        public const string ArchiveEnableValue = "1";
        public const string ArchiveOutputVariable = "WIS_FULL_HAND_REVIEW_ARCHIVE_OUTPUT";

        /// <summary>Test-only environment source. Production uses the real environment.</summary>
        public static Func<string, string> EnvironmentForTest;

        /// <summary>Test-only session. When set, no configuration or bundle is resolved.</summary>
        public static FullHandObservationSession SessionForTest;

        /// <summary>Test-only console sink. Production uses the Unity log.</summary>
        public static Action<string> LogForTest;

        static bool resolved;
        static FullHandObserverState state = FullHandObserverState.NotAttempted;
        static string reasonCode = string.Empty;
        static FullHandObservationSession session;
        static bool archiveEnabled;
        static string archiveReasonCode = string.Empty;

        public static FullHandObserverState State { get { return state; } }
        public static string ReasonCode { get { return reasonCode; } }
        /// <summary>True only when a review-only archive sink was actually resolved.</summary>
        public static bool ArchiveEnabled { get { return archiveEnabled; } }
        /// <summary>Typed archive configuration outcome; never a path.</summary>
        public static string ArchiveReasonCode { get { return archiveReasonCode; } }
        public static int LoadAttempts { get; private set; }
        public static string LastStatusLine { get; private set; }

        public static void ResetForTest()
        {
            resolved = false;
            state = FullHandObserverState.NotAttempted;
            reasonCode = string.Empty;
            session = null;
            archiveEnabled = false;
            archiveReasonCode = string.Empty;
            SessionForTest = null;
            EnvironmentForTest = null;
            LogForTest = null;
            LoadAttempts = 0;
            LastStatusLine = string.Empty;
        }

        /// <summary>
        /// The controller's only entry point. Silent and inert unless deliberately enabled;
        /// every fault, including a missing model or an IO failure, is contained here.
        /// </summary>
        public static void SafeObserve(string captureSessionId, string declaredTarget,
                                       IReadOnlyList<RawMediaPipeObservation> observations)
        {
            try
            {
                FullHandObservationSession active = Resolve();
                if (active == null) return;
                string status;
                active.TryObserve(captureSessionId, declaredTarget, observations, out status);
                LastStatusLine = active.LastStatusLine;
            }
            catch (Exception)
            {
                // read-only diagnostic: the capture path must never see this
            }
        }

        static FullHandObservationSession Resolve()
        {
            if (SessionForTest != null) return SessionForTest;
            if (resolved) return session;
            resolved = true;                       // exactly one resolution per process
            Func<string, string> read = EnvironmentForTest ?? Environment.GetEnvironmentVariable;
            string enabled;
            try { enabled = read(EnableVariable); }
            catch (Exception) { Fault("ENVIRONMENT_UNREADABLE"); return null; }
            if (!string.Equals(enabled, EnableValue, StringComparison.Ordinal))
            {
                state = FullHandObserverState.Disabled;
                reasonCode = "NOT_ENABLED";
                return null;                       // nothing loaded, nothing logged
            }
            string manifest = Safe(read, ManifestVariable);
            string output = Safe(read, OutputVariable);
            // fully qualified is required on the RAW value, before any normalization
            if (!FullHandFileObservationSink.IsFullyQualified(manifest))
            { Fault("BUNDLE_MANIFEST_NOT_ABSOLUTE"); return null; }
            if (!FullHandFileObservationSink.IsFullyQualified(output))
            { Fault("OUTPUT_ROOT_NOT_ABSOLUTE"); return null; }
            try
            {
                LoadAttempts++;
                FullHandRuntimeBundle bundle = FullHandRuntimeBundle.Load(manifest);
                var sink = new FullHandFileObservationSink(output);
                // resolved only now: an archive variable is never read while the observer
                // itself is disabled or faulted, and a bad archive root disables the
                // archive alone, leaving the observer Ready
                session = new FullHandObservationSession(bundle, sink, Write,
                                                         ResolveArchive(read));
                state = FullHandObserverState.Ready;
                reasonCode = string.Empty;
                Write("[FullHandObserve] setup=READY refs=" + bundle.ReferenceCount
                      + " classes=" + bundle.VocabularyCount
                      + " note=opt_in_read_only_diagnostic.no_promotion");
                return session;
            }
            catch (FullHandFormatException error) { Fault(error.Code); }
            catch (Exception error) { Fault("SETUP_" + error.GetType().Name); }
            return null;
        }

        /// <summary>
        /// OFF unless deliberately enabled. Returns null for every disabled or invalid
        /// configuration; the caller stays Ready either way. No default root exists.
        /// </summary>
        static IFullHandReviewCandidateSink ResolveArchive(Func<string, string> read)
        {
            string enabled = Safe(read, ArchiveEnableVariable);
            if (!string.Equals(enabled, ArchiveEnableValue, StringComparison.Ordinal))
            {
                archiveReasonCode = "NOT_ENABLED";   // no root read, nothing logged
                return null;
            }
            string root = Safe(read, ArchiveOutputVariable);
            if (!FullHandFileObservationSink.IsFullyQualified(root))
            { ArchiveFault("ARCHIVE_ROOT_NOT_ABSOLUTE"); return null; }
            try
            {
                var archive = new FullHandFileReviewCandidateSink(root);
                archiveEnabled = true;
                archiveReasonCode = string.Empty;
                Write("[FullHandArchive] setup=READY note=review_only.not_enrollment");
                return archive;
            }
            catch (FullHandFormatException error) { ArchiveFault(error.Code); }
            catch (Exception error) { ArchiveFault("SETUP_" + error.GetType().Name); }
            return null;
        }

        static void ArchiveFault(string code)
        {
            archiveEnabled = false;
            archiveReasonCode = code;
            // one short typed line; the observer itself is unaffected
            Write("[FullHandArchive] setup=UNAVAILABLE reason=" + code);
        }

        static string Safe(Func<string, string> read, string name)
        {
            try { return read(name); }
            catch (Exception) { return null; }
        }

        static void Fault(string code)
        {
            state = FullHandObserverState.Faulted;
            reasonCode = code;
            session = null;
            // one short typed line, carrying no path and no identifier
            Write("[FullHandObserve] setup=UNAVAILABLE reason=" + code);
        }

        static void Write(string line)
        {
            LastStatusLine = line;
            Action<string> sink = LogForTest;
            try
            {
                if (sink != null) sink(line);
                else UnityEngine.Debug.Log(line);
            }
            catch (Exception)
            {
                // a failing log must never reach the capture path
            }
        }
    }
}
