// F11 - full-hand REVIEW-CANDIDATE archive. Additional API only.
//
// Scope and non-scope (read before wiring anything to this file):
//   * This is storage groundwork. It does NOT enroll, promote, approve, train,
//     score, threshold, or recognize anything, and it contains no word-specific
//     branch of any kind.
//   * Every record produced or read here is ReviewOnly and IsEnrollmentEligible
//     is hard-wired to false, including records whose quality state is
//     Confirmed. There is deliberately no approval / promotion API.
//   * DeclaredTargetWord is caller-declared metadata only. A predicted word that
//     happens to equal it is NEVER treated as evidence that the sample is
//     correct; nothing in this file compares those two strings.
//   * The SHA-256 in the envelope is CORRUPTION DETECTION, not authenticity: the
//     digest is computed over plaintext with no secret, so anyone able to edit
//     the file can recompute it. ModelIdentity and quality EvidenceId are
//     free-form PROVENANCE LABELS supplied by the caller; storing them is not
//     proof that the named model or assessor produced the record.
//   * Nothing here reads an environment variable, uses a default directory,
//     touches Unity, migrates existing data, scans a tree, or overwrites a file.
//   * The 46-point schema is preserved whole. Missing joints are never filled in
//     and the frozen 16-point format is never produced or consumed.
//
// C# 5 compatible; depends only on the System BCL and FullHandCommonCapture.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.Personalization
{
    /// <summary>Raised for any malformed / corrupt / unsupported archive bytes.</summary>
    public sealed class FullHandArchiveFormatException : Exception
    {
        public FullHandArchiveFormatException(string message) : base(message) { }
        public FullHandArchiveFormatException(string message, Exception inner)
            : base(message, inner) { }
    }

    /// <summary>Assessor-side quality verdict. Carries no enrollment authority.</summary>
    public enum FullHandReviewQualityState
    {
        Unassessed = 0,
        Rejected = 1,
        Confirmed = 2,
    }

    /// <summary>
    /// Word-side evidence DTO, kept strictly separate from quality evidence.
    /// Either the record is unscored (no predicted word, no score, no model
    /// identity, no feature schema) or it is scored and all four are present.
    /// The score is an opaque finite number: this file assigns it NO probability
    /// semantics, no range, no direction and no threshold.
    /// </summary>
    public sealed class FullHandWordEvidence
    {
        readonly bool scored;
        readonly string predictedWord;
        readonly double wordScore;
        readonly string modelIdentity;
        readonly string featureSchemaId;

        FullHandWordEvidence(bool scored, string predictedWord, double wordScore,
                             string modelIdentity, string featureSchemaId)
        {
            this.scored = scored;
            this.predictedWord = predictedWord;
            this.wordScore = wordScore;
            this.modelIdentity = modelIdentity;
            this.featureSchemaId = featureSchemaId;
        }

        public bool IsScored { get { return scored; } }
        /// <summary>Empty when unscored.</summary>
        public string PredictedWord { get { return predictedWord; } }
        /// <summary>Meaningful only when IsScored; 0.0 placeholder otherwise.</summary>
        public double WordScore { get { return wordScore; } }
        /// <summary>Empty when unscored, else a 64-char lowercase SHA-256 hex label.</summary>
        public string ModelIdentity { get { return modelIdentity; } }
        /// <summary>Empty when unscored, else the non-empty feature schema id.</summary>
        public string FeatureSchemaId { get { return featureSchemaId; } }

        public static FullHandWordEvidence Unscored()
        {
            return new FullHandWordEvidence(false, string.Empty, 0.0,
                                            string.Empty, string.Empty);
        }

        public static FullHandWordEvidence Scored(string predictedWord, double wordScore,
                                                  string modelIdentitySha256Hex,
                                                  string featureSchemaId)
        {
            ArchiveText.RequireLabel(predictedWord, "predictedWord");
            if (double.IsNaN(wordScore) || double.IsInfinity(wordScore))
                throw new ArgumentOutOfRangeException("wordScore", "wordScore must be finite.");
            ArchiveText.RequireSha256Hex(modelIdentitySha256Hex, "modelIdentitySha256Hex");
            ArchiveText.RequireLabel(featureSchemaId, "featureSchemaId");
            return new FullHandWordEvidence(true, predictedWord, wordScore,
                                            modelIdentitySha256Hex.ToLowerInvariant(),
                                            featureSchemaId);
        }

        internal static FullHandWordEvidence FromDecoded(bool scored, string predictedWord,
            double wordScore, string modelIdentity, string featureSchemaId)
        {
            if (scored)
                return Scored(predictedWord, wordScore, modelIdentity, featureSchemaId);
            if (predictedWord.Length != 0 || modelIdentity.Length != 0 ||
                featureSchemaId.Length != 0 || wordScore != 0.0)
                throw new FullHandArchiveFormatException(
                    "Unscored word evidence must carry empty provenance and a 0.0 score.");
            return Unscored();
        }
    }

    /// <summary>
    /// Quality-side evidence DTO. Independent of word evidence: a record may be
    /// Confirmed while unscored, or Rejected while scored. EvidenceId is required
    /// for Confirmed, optional for Rejected, and must be empty for Unassessed.
    /// </summary>
    public sealed class FullHandQualityEvidence
    {
        readonly FullHandReviewQualityState state;
        readonly string evidenceId;

        FullHandQualityEvidence(FullHandReviewQualityState state, string evidenceId)
        {
            this.state = state;
            this.evidenceId = evidenceId;
        }

        public FullHandReviewQualityState State { get { return state; } }
        public string EvidenceId { get { return evidenceId; } }

        public static FullHandQualityEvidence Unassessed()
        {
            return new FullHandQualityEvidence(FullHandReviewQualityState.Unassessed,
                                               string.Empty);
        }

        public static FullHandQualityEvidence Rejected(string evidenceId)
        {
            return Create(FullHandReviewQualityState.Rejected, evidenceId);
        }

        public static FullHandQualityEvidence Confirmed(string evidenceId)
        {
            return Create(FullHandReviewQualityState.Confirmed, evidenceId);
        }

        /// <summary>Confirmed here means "quality looks usable to a reviewer".
        /// It never grants enrollment eligibility.</summary>
        public static FullHandQualityEvidence Create(FullHandReviewQualityState state,
                                                     string evidenceId)
        {
            if (state != FullHandReviewQualityState.Unassessed &&
                state != FullHandReviewQualityState.Rejected &&
                state != FullHandReviewQualityState.Confirmed)
                throw new ArgumentOutOfRangeException("state");
            string id = evidenceId ?? string.Empty;
            if (id.Length != 0) ArchiveText.RequireLabel(id, "evidenceId");
            if (state == FullHandReviewQualityState.Confirmed && id.Length == 0)
                throw new ArgumentException("Confirmed requires a non-empty evidenceId.",
                                            "evidenceId");
            if (state == FullHandReviewQualityState.Unassessed && id.Length != 0)
                throw new ArgumentException("Unassessed must not carry an evidenceId.",
                                            "evidenceId");
            return new FullHandQualityEvidence(state, id);
        }
    }

    /// <summary>
    /// Immutable review-only candidate: a cloned 46-point capture, its cloned
    /// per-frame timestamps, an anonymous sample id, the declared target word,
    /// and the two independent evidence DTOs. Construction fully validates; the
    /// codec re-validates on both serialize and deserialize.
    /// </summary>
    public sealed class FullHandReviewCandidate
    {
        public const int MaxFrameCount = 4096;
        public const int MaxSampleIdLength = 64;
        public const string RecordSchemaId = "wis_full_hand_review_candidate_v1";

        readonly FullHandCommonCapture capture;
        readonly double[] timestamps;
        readonly string sampleId;
        readonly string declaredTargetWord;
        readonly FullHandWordEvidence wordEvidence;
        readonly FullHandQualityEvidence qualityEvidence;

        public FullHandReviewCandidate(FullHandCommonCapture capture, double[] timestamps,
                                       string sampleId, string declaredTargetWord,
                                       FullHandWordEvidence wordEvidence,
                                       FullHandQualityEvidence qualityEvidence)
        {
            if (capture == null) throw new ArgumentNullException("capture");
            if (timestamps == null) throw new ArgumentNullException("timestamps");
            if (wordEvidence == null) throw new ArgumentNullException("wordEvidence");
            if (qualityEvidence == null) throw new ArgumentNullException("qualityEvidence");
            RequireSampleId(sampleId);
            ArchiveText.RequireLabel(declaredTargetWord, "declaredTargetWord");

            // Clone the capture through its own validating constructor so the record
            // owns its arrays and later caller mutation cannot reach them.
            this.capture = CloneCapture(capture);
            this.timestamps = (double[])timestamps.Clone();
            this.sampleId = sampleId;
            this.declaredTargetWord = declaredTargetWord;
            this.wordEvidence = wordEvidence;
            this.qualityEvidence = qualityEvidence;

            FullHandObservationBinary.ValidateCapture(this.capture, this.timestamps);
        }

        /// <summary>Hard-wired: this archive never yields an enrollable sample.</summary>
        public bool IsReviewOnly { get { return true; } }
        /// <summary>Hard-wired false, for every quality state including Confirmed.</summary>
        public bool IsEnrollmentEligible { get { return false; } }

        /// <summary>The owned capture (FullHandCommonCapture is itself immutable).</summary>
        public FullHandCommonCapture Capture { get { return capture; } }
        public int FrameCount { get { return capture.FrameCount; } }
        public string SampleId { get { return sampleId; } }
        public string DeclaredTargetWord { get { return declaredTargetWord; } }
        public FullHandWordEvidence WordEvidence { get { return wordEvidence; } }
        public FullHandQualityEvidence QualityEvidence { get { return qualityEvidence; } }
        public double[] CopyTimestamps() { return (double[])timestamps.Clone(); }

        /// <summary>Immutable copy with different word evidence. Not a promotion.</summary>
        public FullHandReviewCandidate WithWordEvidence(FullHandWordEvidence replacement)
        {
            return new FullHandReviewCandidate(capture, timestamps, sampleId,
                                               declaredTargetWord, replacement,
                                               qualityEvidence);
        }

        /// <summary>Immutable copy with a different quality verdict. Not a promotion.</summary>
        public FullHandReviewCandidate WithQualityEvidence(FullHandQualityEvidence replacement)
        {
            return new FullHandReviewCandidate(capture, timestamps, sampleId,
                                               declaredTargetWord, wordEvidence, replacement);
        }

        /// <summary>Immutable copy under a different anonymous sample id.</summary>
        public FullHandReviewCandidate WithSampleId(string replacementSampleId)
        {
            return new FullHandReviewCandidate(capture, timestamps, replacementSampleId,
                                               declaredTargetWord, wordEvidence,
                                               qualityEvidence);
        }

        /// <summary>
        /// The bare WISFHOB1 observation payload for this record, so the existing
        /// writer can be byte-compared without pulling in Unity.
        /// </summary>
        public byte[] ToObservationBinary()
        {
            return FullHandObservationBinary.Write(capture, timestamps);
        }

        internal static FullHandCommonCapture CloneCapture(FullHandCommonCapture source)
        {
            return new FullHandCommonCapture(
                source.FrameCount, source.CopyCoordinates(), source.CopyValidMask(),
                source.CopyOriginalValidMask(), source.CopyInterpolatedMask(),
                source.ValidPointCount, source.InterpolatedPointCount,
                source.ImageWidth, source.ImageHeight, source.ConventionId);
        }

        /// <summary>Anonymous id: ASCII letter/digit/underscore/hyphen, 1..64 chars.</summary>
        public static bool IsValidSampleId(string sampleId)
        {
            if (string.IsNullOrEmpty(sampleId)) return false;
            if (sampleId.Length > MaxSampleIdLength) return false;
            for (int i = 0; i < sampleId.Length; i++)
            {
                char c = sampleId[i];
                bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '_' || c == '-';
                if (!ok) return false;
            }
            return true;
        }

        internal static void RequireSampleId(string sampleId)
        {
            if (sampleId == null) throw new ArgumentNullException("sampleId");
            if (!IsValidSampleId(sampleId))
                throw new ArgumentException(
                    "sampleId must be 1..64 ASCII letters, digits, '_' or '-'.", "sampleId");
        }
    }

    /// <summary>
    /// The WISFHOB1 observation payload: the pre-existing on-wire layout, preserved
    /// byte for byte.
    ///
    ///   [0..7]    ASCII magic "WISFHOB1"
    ///   [8..11]   int32LE payload version = 1
    ///   [12..15]  int32LE T   (frame count, 1..4096)
    ///   [16..19]  int32LE 46  (point count)
    ///   [20..23]  int32LE 3   (coordinate dimensions)
    ///   [24..]    T   * float64LE timestamps
    ///             T*46*3 float64LE coordinates, flattened frame-major then point
    ///                    then axis
    ///             T*46   bytes final normalized valid mask   (0 or 1)
    ///             T*46   bytes original-valid (source) mask  (0 or 1)
    ///             T*46   bytes interpolated (source) mask    (0 or 1)
    ///   total = 24 + 1250*T bytes, no trailing bytes permitted.
    /// </summary>
    public static class FullHandObservationBinary
    {
        public const string Magic = "WISFHOB1";
        public const int Version = 1;
        const int Points = FullHandCommonSchema.PointCount;        // 46
        const int Axes = FullHandCommonSchema.CoordinateDimensions; // 3

        public static int ByteLength(int frameCount)
        {
            if (frameCount <= 0 || frameCount > FullHandReviewCandidate.MaxFrameCount)
                throw new ArgumentOutOfRangeException("frameCount");
            checked
            {
                int slots = frameCount * Points;
                return 24 + (frameCount * 8) + (slots * Axes * 8) + (slots * 3);
            }
        }

        /// <summary>
        /// Full structural validation of a capture plus its timestamps. Throws
        /// ArgumentException / ArgumentOutOfRangeException on any violation.
        /// </summary>
        public static void ValidateCapture(FullHandCommonCapture capture, double[] timestamps)
        {
            if (capture == null) throw new ArgumentNullException("capture");
            if (timestamps == null) throw new ArgumentNullException("timestamps");

            int frames = capture.FrameCount;
            if (frames <= 0 || frames > FullHandReviewCandidate.MaxFrameCount)
                throw new ArgumentOutOfRangeException("capture",
                    "frameCount must be 1.." + FullHandReviewCandidate.MaxFrameCount + ".");
            if (capture.PointCount != Points)
                throw new ArgumentException("Capture must carry all 46 points.", "capture");
            if (capture.SchemaId != FullHandCommonSchema.SchemaId)
                throw new ArgumentException("Unexpected capture schema id.", "capture");
            if (timestamps.Length != frames)
                throw new ArgumentException("timestamps length must equal frameCount.",
                                            "timestamps");
            if (capture.ImageWidth <= 0 || capture.ImageHeight <= 0)
                throw new ArgumentOutOfRangeException("capture",
                    "Image width and height must be positive.");
            if (string.IsNullOrEmpty(capture.ConventionId))
                throw new ArgumentException("ConventionId must be non-empty.", "capture");
            ArchiveText.RequireLabel(capture.ConventionId, "capture.ConventionId");
            ArchiveText.RequireLabel(capture.DomainContract, "capture.DomainContract");

            for (int t = 0; t < frames; t++)
            {
                double value = timestamps[t];
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new ArgumentException("Timestamps must be finite.", "timestamps");
                if (t > 0 && !(value > timestamps[t - 1]))
                    throw new ArgumentException("Timestamps must strictly increase.",
                                                "timestamps");
            }

            double[] coordinates = capture.CopyCoordinates();
            bool[] finalMask = capture.CopyValidMask();
            bool[] originalMask = capture.CopyOriginalValidMask();
            bool[] interpolatedMask = capture.CopyInterpolatedMask();

            // Count semantics are derived from the production writer
            // (ApprovedLivePreprocessingAdapter.WriteComponent), not assumed:
            //   ValidPointCount        = number of FINAL-normalized-valid points
            //                            (interpolated points included).
            //   InterpolatedPointCount = number of points that are BOTH final-valid
            //                            AND interpolated in the source.
            // The raw source-mask totals are deliberately NOT compared to either
            // counter: the source masks keep earlier provenance even where the final
            // mask is false, so raw totals legitimately differ from the counters.
            int finalCount = 0, finalInterpolatedCount = 0;
            for (int slot = 0; slot < finalMask.Length; slot++)
            {
                bool isFinal = finalMask[slot];
                bool wasOriginal = originalMask[slot];
                bool wasInterpolated = interpolatedMask[slot];

                // Source masks are disjoint.
                if (wasOriginal && wasInterpolated)
                    throw new ArgumentException(
                        "originalValid and interpolated must be disjoint at slot " + slot,
                        "capture");
                // final => original OR interpolated. The converse is NOT required:
                // an originally valid point may end up finally invalid when torso
                // normalization was unavailable for its frame.
                if (isFinal && !(wasOriginal || wasInterpolated))
                    throw new ArgumentException(
                        "Final-valid slot " + slot + " has no source evidence.", "capture");

                if (isFinal)
                {
                    finalCount++;
                    if (wasInterpolated) finalInterpolatedCount++;
                }

                int at = slot * Axes;
                double x = coordinates[at], y = coordinates[at + 1], z = coordinates[at + 2];
                if (isFinal)
                {
                    if (double.IsNaN(x) || double.IsInfinity(x) ||
                        double.IsNaN(y) || double.IsInfinity(y))
                        throw new ArgumentException(
                            "Valid slot " + slot + " must hold finite x/y.", "capture");
                    if (z != 0.0)
                        throw new ArgumentException(
                            "Valid slot " + slot + " must hold z exactly 0.", "capture");
                }
                else
                {
                    if (!double.IsNaN(x) || !double.IsNaN(y) || !double.IsNaN(z))
                        throw new ArgumentException(
                            "Masked slot " + slot + " must hold NaN in all channels.",
                            "capture");
                }
            }

            // The declared counters must match the production definitions exactly.
            if (capture.ValidPointCount != finalCount)
                throw new ArgumentException(
                    "ValidPointCount " + capture.ValidPointCount +
                    " disagrees with the final-valid mask count " + finalCount + ".",
                    "capture");
            if (capture.InterpolatedPointCount != finalInterpolatedCount)
                throw new ArgumentException(
                    "InterpolatedPointCount " + capture.InterpolatedPointCount +
                    " disagrees with the final-valid AND interpolated count " +
                    finalInterpolatedCount + ".", "capture");
            // A fully empty final mask stays legal (both counters are then 0): it is
            // review evidence of a bad take. It is still never enrollment evidence.
        }

        /// <summary>Serializes the capture to the exact WISFHOB1 layout.</summary>
        public static byte[] Write(FullHandCommonCapture capture, double[] timestamps)
        {
            ValidateCapture(capture, timestamps);
            int frames = capture.FrameCount;
            byte[] buffer = new byte[ByteLength(frames)];
            int at = 0;
            for (int i = 0; i < 8; i++) buffer[at++] = (byte)Magic[i];
            ArchiveBytes.PutInt32(buffer, ref at, Version);
            ArchiveBytes.PutInt32(buffer, ref at, frames);
            ArchiveBytes.PutInt32(buffer, ref at, Points);
            ArchiveBytes.PutInt32(buffer, ref at, Axes);

            for (int t = 0; t < frames; t++)
                ArchiveBytes.PutDouble(buffer, ref at, timestamps[t]);

            double[] coordinates = capture.CopyCoordinates();
            for (int i = 0; i < coordinates.Length; i++)
                ArchiveBytes.PutDouble(buffer, ref at, coordinates[i]);

            PutMask(buffer, ref at, capture.CopyValidMask());
            PutMask(buffer, ref at, capture.CopyOriginalValidMask());
            PutMask(buffer, ref at, capture.CopyInterpolatedMask());
            if (at != buffer.Length)
                throw new InvalidOperationException("Observation writer length mismatch.");
            return buffer;
        }

        static void PutMask(byte[] buffer, ref int at, bool[] mask)
        {
            for (int i = 0; i < mask.Length; i++) buffer[at++] = mask[i] ? (byte)1 : (byte)0;
        }

        /// <summary>
        /// Parses a standalone WISFHOB1 payload. The capture-level counts and image
        /// metadata are not in the payload, so they are supplied by the envelope.
        /// </summary>
        public static FullHandCommonCapture Read(byte[] payload, int offset, int length,
                                                 int validPointCount, int interpolatedPointCount,
                                                 int imageWidth, int imageHeight,
                                                 string conventionId, out double[] timestamps)
        {
            if (payload == null) throw new ArgumentNullException("payload");
            if (offset < 0 || length < 0 || offset > payload.Length - length)
                throw new ArgumentOutOfRangeException("length");
            if (length < 24)
                throw new FullHandArchiveFormatException("Observation payload is truncated.");

            int at = offset;
            for (int i = 0; i < 8; i++)
                if (payload[at + i] != (byte)Magic[i])
                    throw new FullHandArchiveFormatException("Bad observation magic.");
            at += 8;
            int version = ArchiveBytes.GetInt32(payload, ref at);
            int frames = ArchiveBytes.GetInt32(payload, ref at);
            int points = ArchiveBytes.GetInt32(payload, ref at);
            int axes = ArchiveBytes.GetInt32(payload, ref at);
            if (version != Version)
                throw new FullHandArchiveFormatException(
                    "Unsupported observation version " + version + ".");
            if (points != Points || axes != Axes)
                throw new FullHandArchiveFormatException(
                    "Observation must be 46 points x 3 axes; the 16-point format is " +
                    "never accepted.");
            if (frames <= 0 || frames > FullHandReviewCandidate.MaxFrameCount)
                throw new FullHandArchiveFormatException(
                    "Observation frame count " + frames + " is out of range.");
            if (length != ByteLength(frames))
                throw new FullHandArchiveFormatException(
                    "Observation payload length " + length + " does not match frame count " +
                    frames + ".");

            double[] times = new double[frames];
            for (int t = 0; t < frames; t++)
                times[t] = ArchiveBytes.GetDouble(payload, ref at);

            int slots = frames * Points;
            double[] coordinates = new double[slots * Axes];
            for (int i = 0; i < coordinates.Length; i++)
                coordinates[i] = ArchiveBytes.GetDouble(payload, ref at);

            bool[] finalMask = GetMask(payload, ref at, slots);
            bool[] originalMask = GetMask(payload, ref at, slots);
            bool[] interpolatedMask = GetMask(payload, ref at, slots);
            if (at != offset + length)
                throw new FullHandArchiveFormatException(
                    "Observation payload has trailing bytes.");

            FullHandCommonCapture capture;
            try
            {
                capture = new FullHandCommonCapture(frames, coordinates, finalMask,
                                                    originalMask, interpolatedMask,
                                                    validPointCount, interpolatedPointCount,
                                                    imageWidth, imageHeight, conventionId);
                ValidateCapture(capture, times);
            }
            catch (ArgumentException error)
            {
                throw new FullHandArchiveFormatException(
                    "Decoded observation failed validation: " + error.Message, error);
            }
            timestamps = times;
            return capture;
        }

        static bool[] GetMask(byte[] payload, ref int at, int count)
        {
            bool[] mask = new bool[count];
            for (int i = 0; i < count; i++)
            {
                byte raw = payload[at++];
                if (raw > 1)
                    throw new FullHandArchiveFormatException(
                        "Mask byte " + i + " is " + raw + "; only 0 or 1 is legal.");
                mask[i] = raw == 1;
            }
            return mask;
        }
    }

    /// <summary>
    /// The WISFHC01 archive envelope. Deterministic: identical inputs always
    /// produce identical bytes.
    ///
    /// FILE
    ///   [0..7]    ASCII magic "WISFHC01"
    ///   [8..11]   int32LE archive version = 1
    ///   [12..15]  int32LE bodyLength
    ///   [16..47]  32 raw bytes SHA-256(body)     - corruption detection only
    ///   [48..]    body, exactly bodyLength bytes; the file must end there.
    ///
    /// BODY (str = int32LE UTF-8 byte count 0..1024, then that many UTF-8 bytes)
    ///    1  str    recordSchemaId = "wis_full_hand_review_candidate_v1"
    ///    2  str    sampleId
    ///    3  str    declaredTargetWord
    ///    4  byte   wordScored (0 or 1)
    ///    5  str    predictedWord            (empty unless scored)
    ///    6  f64LE  wordScore                (0.0 unless scored; finite when scored)
    ///    7  str    modelIdentity            (empty unless scored, else sha256 hex)
    ///    8  str    featureSchemaId          (empty unless scored)
    ///    9  int32LE qualityState  0=Unassessed 1=Rejected 2=Confirmed
    ///   10  str    qualityEvidenceId        (non-empty iff Confirmed; empty if Unassessed)
    ///   11  str    captureSchemaId = "wis_full_hand_common_v1"
    ///   12  str    domainContract
    ///   13  str    conventionId
    ///   14  int32LE imageWidth   (&gt; 0)
    ///   15  int32LE imageHeight  (&gt; 0)
    ///   16  int32LE validPointCount        = count(final mask true)
    ///   17  int32LE interpolatedPointCount = count(final true AND interpolated true)
    ///   18  int32LE observationPayloadLength
    ///   19  bytes  observationPayload (WISFHOB1, see FullHandObservationBinary)
    ///   The body must end exactly after the payload.
    ///
    /// The digest covers the metadata AND the payload, so an edited label is
    /// detected just like an edited coordinate. It is not authentication.
    /// </summary>
    public static class FullHandReviewCandidateCodec
    {
        public const string Magic = "WISFHC01";
        public const int Version = 1;
        public const int HeaderLength = 48;
        public const int DigestLength = 32;
        public const int MaxArchiveBytes = 16 * 1024 * 1024;
        public const int MaxBodyBytes = MaxArchiveBytes - HeaderLength;

        public static byte[] Serialize(FullHandReviewCandidate candidate)
        {
            if (candidate == null) throw new ArgumentNullException("candidate");
            // Re-validate rather than trust the in-memory record.
            double[] timestamps = candidate.CopyTimestamps();
            FullHandCommonCapture capture = candidate.Capture;
            FullHandObservationBinary.ValidateCapture(capture, timestamps);
            FullHandReviewCandidate.RequireSampleId(candidate.SampleId);
            ArchiveText.RequireLabel(candidate.DeclaredTargetWord, "declaredTargetWord");

            FullHandWordEvidence word = candidate.WordEvidence;
            FullHandQualityEvidence quality = candidate.QualityEvidence;
            // Re-assert both DTO invariants through their validating factories.
            FullHandWordEvidence.FromDecoded(word.IsScored, word.PredictedWord, word.WordScore,
                                             word.ModelIdentity, word.FeatureSchemaId);
            FullHandQualityEvidence.Create(quality.State, quality.EvidenceId);

            byte[] payload = FullHandObservationBinary.Write(capture, timestamps);

            var body = new List<byte>(payload.Length + 1024);
            ArchiveBytes.AddString(body, FullHandReviewCandidate.RecordSchemaId);
            ArchiveBytes.AddString(body, candidate.SampleId);
            ArchiveBytes.AddString(body, candidate.DeclaredTargetWord);
            body.Add(word.IsScored ? (byte)1 : (byte)0);
            ArchiveBytes.AddString(body, word.PredictedWord);
            ArchiveBytes.AddDouble(body, word.IsScored ? word.WordScore : 0.0);
            ArchiveBytes.AddString(body, word.ModelIdentity);
            ArchiveBytes.AddString(body, word.FeatureSchemaId);
            ArchiveBytes.AddInt32(body, (int)quality.State);
            ArchiveBytes.AddString(body, quality.EvidenceId);
            ArchiveBytes.AddString(body, capture.SchemaId);
            ArchiveBytes.AddString(body, capture.DomainContract);
            ArchiveBytes.AddString(body, capture.ConventionId);
            ArchiveBytes.AddInt32(body, capture.ImageWidth);
            ArchiveBytes.AddInt32(body, capture.ImageHeight);
            ArchiveBytes.AddInt32(body, capture.ValidPointCount);
            ArchiveBytes.AddInt32(body, capture.InterpolatedPointCount);
            ArchiveBytes.AddInt32(body, payload.Length);
            body.AddRange(payload);

            byte[] bodyBytes = body.ToArray();
            if (bodyBytes.Length > MaxBodyBytes)
                throw new ArgumentException(
                    "Archive body " + bodyBytes.Length + " exceeds the " + MaxBodyBytes +
                    " byte limit.", "candidate");

            int total;
            checked { total = HeaderLength + bodyBytes.Length; }
            byte[] file = new byte[total];
            int at = 0;
            for (int i = 0; i < 8; i++) file[at++] = (byte)Magic[i];
            ArchiveBytes.PutInt32(file, ref at, Version);
            ArchiveBytes.PutInt32(file, ref at, bodyBytes.Length);
            byte[] digest = ArchiveBytes.Sha256(bodyBytes);
            Buffer.BlockCopy(digest, 0, file, at, DigestLength);
            at += DigestLength;
            Buffer.BlockCopy(bodyBytes, 0, file, at, bodyBytes.Length);
            return file;
        }

        public static FullHandReviewCandidate Deserialize(byte[] archiveBytes)
        {
            if (archiveBytes == null) throw new ArgumentNullException("archiveBytes");
            if (archiveBytes.Length > MaxArchiveBytes)
                throw new FullHandArchiveFormatException(
                    "Archive is larger than the " + MaxArchiveBytes + " byte limit.");
            if (archiveBytes.Length < HeaderLength)
                throw new FullHandArchiveFormatException("Archive header is truncated.");
            for (int i = 0; i < 8; i++)
                if (archiveBytes[i] != (byte)Magic[i])
                    throw new FullHandArchiveFormatException(
                        "Bad archive magic (a bare WISFHOB1 payload is not an archive).");
            int at = 8;
            int version = ArchiveBytes.GetInt32(archiveBytes, ref at);
            if (version != Version)
                throw new FullHandArchiveFormatException(
                    "Unsupported archive version " + version + ".");
            int bodyLength = ArchiveBytes.GetInt32(archiveBytes, ref at);
            if (bodyLength <= 0 || bodyLength > MaxBodyBytes)
                throw new FullHandArchiveFormatException(
                    "Declared body length " + bodyLength + " is out of range.");
            if (archiveBytes.Length != HeaderLength + bodyLength)
                throw new FullHandArchiveFormatException(
                    "Archive length " + archiveBytes.Length + " does not match the declared " +
                    "body length (truncated or trailing bytes).");

            byte[] body = new byte[bodyLength];
            Buffer.BlockCopy(archiveBytes, HeaderLength, body, 0, bodyLength);
            // Verify the digest BEFORE interpreting any of the body.
            byte[] actual = ArchiveBytes.Sha256(body);
            for (int i = 0; i < DigestLength; i++)
                if (actual[i] != archiveBytes[16 + i])
                    throw new FullHandArchiveFormatException(
                        "Archive digest mismatch: the body is corrupt or was edited.");

            int b = 0;
            string recordSchema = ArchiveBytes.GetString(body, ref b);
            if (recordSchema != FullHandReviewCandidate.RecordSchemaId)
                throw new FullHandArchiveFormatException(
                    "Unsupported record schema '" + recordSchema + "'.");
            string sampleId = ArchiveBytes.GetString(body, ref b);
            string declaredTargetWord = ArchiveBytes.GetString(body, ref b);
            byte scoredFlag = ArchiveBytes.GetByte(body, ref b);
            if (scoredFlag > 1)
                throw new FullHandArchiveFormatException("Invalid wordScored flag.");
            string predictedWord = ArchiveBytes.GetString(body, ref b);
            double wordScore = ArchiveBytes.GetDouble(body, ref b);
            string modelIdentity = ArchiveBytes.GetString(body, ref b);
            string featureSchemaId = ArchiveBytes.GetString(body, ref b);
            int qualityRaw = ArchiveBytes.GetInt32(body, ref b);
            if (qualityRaw < 0 || qualityRaw > 2)
                throw new FullHandArchiveFormatException(
                    "Invalid quality state " + qualityRaw + ".");
            string qualityEvidenceId = ArchiveBytes.GetString(body, ref b);
            string captureSchemaId = ArchiveBytes.GetString(body, ref b);
            if (captureSchemaId != FullHandCommonSchema.SchemaId)
                throw new FullHandArchiveFormatException(
                    "Unsupported capture schema '" + captureSchemaId + "'.");
            string domainContract = ArchiveBytes.GetString(body, ref b);
            string conventionId = ArchiveBytes.GetString(body, ref b);
            int imageWidth = ArchiveBytes.GetInt32(body, ref b);
            int imageHeight = ArchiveBytes.GetInt32(body, ref b);
            int validPointCount = ArchiveBytes.GetInt32(body, ref b);
            int interpolatedPointCount = ArchiveBytes.GetInt32(body, ref b);
            int payloadLength = ArchiveBytes.GetInt32(body, ref b);
            if (imageWidth <= 0 || imageHeight <= 0)
                throw new FullHandArchiveFormatException("Image dimensions must be positive.");
            if (validPointCount < 0 || interpolatedPointCount < 0)
                throw new FullHandArchiveFormatException("Point counts must not be negative.");
            if (payloadLength <= 0 || payloadLength != bodyLength - b)
                throw new FullHandArchiveFormatException(
                    "Observation payload length " + payloadLength +
                    " does not fill the remaining body.");

            double[] timestamps;
            FullHandCommonCapture capture = FullHandObservationBinary.Read(
                body, b, payloadLength, validPointCount, interpolatedPointCount,
                imageWidth, imageHeight, conventionId, out timestamps);
            if (capture.DomainContract != domainContract)
                throw new FullHandArchiveFormatException(
                    "Domain contract '" + domainContract + "' is inconsistent with the schema.");

            FullHandWordEvidence word;
            FullHandQualityEvidence quality;
            FullHandReviewCandidate candidate;
            try
            {
                word = FullHandWordEvidence.FromDecoded(scoredFlag == 1, predictedWord,
                                                        wordScore, modelIdentity,
                                                        featureSchemaId);
                quality = FullHandQualityEvidence.Create(
                    (FullHandReviewQualityState)qualityRaw, qualityEvidenceId);
                candidate = new FullHandReviewCandidate(capture, timestamps, sampleId,
                                                        declaredTargetWord, word, quality);
            }
            catch (ArgumentException error)
            {
                throw new FullHandArchiveFormatException(
                    "Decoded record failed validation: " + error.Message, error);
            }
            return candidate;
        }
    }

    /// <summary>
    /// Explicit opt-in storage. The caller must supply an existing, FULLY
    /// QUALIFIED directory every single time (see RequireFullyQualified: Windows
    /// drive-relative and root-relative forms are refused): there is no default
    /// root, no search, no recursion, no overwrite and no migration of any
    /// existing data.
    /// </summary>
    public static class FullHandReviewCandidateStore
    {
        public const string FileExtension = ".wisfhc";

        /// <summary>
        /// Writes the candidate to &lt;directory&gt;/&lt;sampleId&gt;.wisfhc via a unique
        /// partial file in the same directory. Throws IOException if that file
        /// already exists; an existing file is never replaced. Returns the path.
        /// </summary>
        public static string Save(FullHandReviewCandidate candidate, string directoryPath)
        {
            if (candidate == null) throw new ArgumentNullException("candidate");
            string directory = RequireUsableDirectory(directoryPath);
            byte[] bytes = FullHandReviewCandidateCodec.Serialize(candidate);
            if (bytes.Length > FullHandReviewCandidateCodec.MaxArchiveBytes)
                throw new ArgumentException("Serialized archive exceeds the size limit.",
                                            "candidate");
            // Decode-verify our own bytes before they touch the disk.
            FullHandReviewCandidateCodec.Deserialize(bytes);

            string finalPath = Path.Combine(directory,
                                            candidate.SampleId + FileExtension);
            if (File.Exists(finalPath))
                throw new IOException("A candidate file already exists at " + finalPath +
                                      "; this API never replaces it.");

            string partialPath = Path.Combine(
                directory,
                candidate.SampleId + "." + Guid.NewGuid().ToString("N") + ".partial");
            bool partialWritten = false;
            try
            {
                using (var stream = new FileStream(partialPath, FileMode.CreateNew,
                                                   FileAccess.Write, FileShare.None))
                {
                    partialWritten = true;
                    stream.Write(bytes, 0, bytes.Length);
                    stream.Flush(true);
                }
                // Move refuses to overwrite, which is the duplicate guard under a race.
                File.Move(partialPath, finalPath);
                partialWritten = false;
            }
            finally
            {
                // Clean up only the partial file this call created.
                if (partialWritten)
                {
                    try { File.Delete(partialPath); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
            }
            return finalPath;
        }

        /// <summary>Loads and fully re-validates one explicitly named .wisfhc file.</summary>
        public static FullHandReviewCandidate Load(string filePath)
        {
            RequireFullyQualified(filePath, "filePath");
            string full = Path.GetFullPath(filePath);
            RequireFullyQualified(full, "filePath");
            RequireNotUnityAssetPath(full);
            if (!string.Equals(Path.GetExtension(full), FileExtension,
                               StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("filePath must end in " + FileExtension + ".",
                                            "filePath");
            var info = new FileInfo(full);
            if (!info.Exists) throw new FileNotFoundException("No archive at " + full, full);
            if (info.Length > FullHandReviewCandidateCodec.MaxArchiveBytes)
                throw new FullHandArchiveFormatException(
                    "Archive file is larger than the " +
                    FullHandReviewCandidateCodec.MaxArchiveBytes + " byte limit.");
            return FullHandReviewCandidateCodec.Deserialize(File.ReadAllBytes(full));
        }

        /// <summary>
        /// The normalized path Save would use. Applies the same fully-qualified,
        /// Assets-refusal and directory-existence checks; writes nothing.
        /// </summary>
        public static string ResolvePath(string directoryPath, string sampleId)
        {
            string directory = RequireUsableDirectory(directoryPath);
            FullHandReviewCandidate.RequireSampleId(sampleId);
            return Path.Combine(directory, sampleId + FileExtension);
        }

        /// <summary>
        /// Validates the directory and returns its normalized absolute form, which
        /// is what every caller must then use.
        /// </summary>
        static string RequireUsableDirectory(string directoryPath)
        {
            RequireFullyQualified(directoryPath, "directoryPath");
            string full = Path.GetFullPath(directoryPath);
            RequireFullyQualified(full, "directoryPath");
            RequireNotUnityAssetPath(full);
            if (!Directory.Exists(full))
                throw new DirectoryNotFoundException(
                    "directoryPath must already exist: " + full);
            return full;
        }

        /// <summary>
        /// Requires a FULLY QUALIFIED path. Path.IsPathRooted is not sufficient on
        /// Windows: it also accepts root-relative ("\Users\x", resolved against the
        /// current drive) and drive-relative ("C:.", "C:work", resolved against that
        /// drive's current directory) forms, both of which depend on ambient process
        /// state. Accepted here:
        ///   Windows - "X:\..." or "X:/...", or a complete UNC "\\server\share..."
        ///             (both separators allowed);
        ///   POSIX   - a leading "/" only.
        /// Rejected: relative, root-relative, drive-relative and URI-style strings.
        /// </summary>
        static void RequireFullyQualified(string path, string name)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentNullException(name);
            if (path.IndexOf("://", StringComparison.Ordinal) >= 0)
                throw new ArgumentException(
                    name + " must be a filesystem path, not a URI: " + path, name);

            bool windows = Path.DirectorySeparatorChar == '\\';
            if (!windows)
            {
                if (path[0] != '/')
                    throw new ArgumentException(
                        name + " must be a fully qualified path beginning with '/': " + path,
                        name);
                return;
            }

            if (path.Length >= 3 && IsAsciiLetter(path[0]) && path[1] == ':' &&
                IsSeparator(path[2]))
                return;                                     // X:\... or X:/...

            if (path.Length >= 2 && IsSeparator(path[0]) && IsSeparator(path[1]))
            {
                // UNC needs a complete server AND share component.
                string[] parts = path.Substring(2).Split(
                    new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
                    StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2) return;
                throw new ArgumentException(
                    name + " must name a complete UNC server and share: " + path, name);
            }

            throw new ArgumentException(
                name + " must be fully qualified (drive-relative and root-relative " +
                "Windows paths are refused): " + path, name);
        }

        static bool IsSeparator(char c)
        {
            return c == Path.DirectorySeparatorChar || c == Path.AltDirectorySeparatorChar;
        }

        static bool IsAsciiLetter(char c)
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }

        static void RequireNotUnityAssetPath(string fullPath)
        {
            string[] segments = fullPath.Split(new[] { Path.DirectorySeparatorChar,
                                                       Path.AltDirectorySeparatorChar },
                                               StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < segments.Length; i++)
            {
                if (string.Equals(segments[i], "Assets", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(segments[i], "StreamingAssets",
                                  StringComparison.OrdinalIgnoreCase))
                    throw new ArgumentException(
                        "Refusing a path under Assets/StreamingAssets: " + fullPath);
            }
        }
    }

    static class ArchiveText
    {
        public const int MaxStringBytes = 1024;
        public static readonly Encoding Utf8Strict = new UTF8Encoding(false, true);

        /// <summary>Non-empty, control-character free, at most 1024 UTF-8 bytes.</summary>
        public static void RequireLabel(string value, string name)
        {
            if (value == null) throw new ArgumentNullException(name);
            if (value.Length == 0)
                throw new ArgumentException(name + " must be non-empty.", name);
            for (int i = 0; i < value.Length; i++)
                if (char.IsControl(value[i]))
                    throw new ArgumentException(name + " must not contain control characters.",
                                                name);
            if (Utf8Strict.GetByteCount(value) > MaxStringBytes)
                throw new ArgumentException(
                    name + " exceeds " + MaxStringBytes + " UTF-8 bytes.", name);
        }

        public static void RequireSha256Hex(string value, string name)
        {
            if (value == null) throw new ArgumentNullException(name);
            if (value.Length != 64)
                throw new ArgumentException(name + " must be 64 hex characters.", name);
            for (int i = 0; i < value.Length; i++)
            {
                char c = char.ToLowerInvariant(value[i]);
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                    throw new ArgumentException(name + " must be hexadecimal.", name);
            }
        }
    }

    static class ArchiveBytes
    {
        public static void PutInt32(byte[] buffer, ref int at, int value)
        {
            buffer[at++] = (byte)(value & 0xFF);
            buffer[at++] = (byte)((value >> 8) & 0xFF);
            buffer[at++] = (byte)((value >> 16) & 0xFF);
            buffer[at++] = (byte)((value >> 24) & 0xFF);
        }

        public static int GetInt32(byte[] buffer, ref int at)
        {
            if (at < 0 || at > buffer.Length - 4)
                throw new FullHandArchiveFormatException("Truncated int32.");
            int value = buffer[at] | (buffer[at + 1] << 8) | (buffer[at + 2] << 16) |
                        (buffer[at + 3] << 24);
            at += 4;
            return value;
        }

        public static void PutDouble(byte[] buffer, ref int at, double value)
        {
            long bits = BitConverter.DoubleToInt64Bits(value);
            for (int i = 0; i < 8; i++) buffer[at++] = (byte)((bits >> (8 * i)) & 0xFF);
        }

        public static double GetDouble(byte[] buffer, ref int at)
        {
            if (at < 0 || at > buffer.Length - 8)
                throw new FullHandArchiveFormatException("Truncated float64.");
            long bits = 0;
            for (int i = 0; i < 8; i++) bits |= ((long)buffer[at + i]) << (8 * i);
            at += 8;
            return BitConverter.Int64BitsToDouble(bits);
        }

        public static byte GetByte(byte[] buffer, ref int at)
        {
            if (at < 0 || at >= buffer.Length)
                throw new FullHandArchiveFormatException("Truncated byte.");
            return buffer[at++];
        }

        public static void AddInt32(List<byte> body, int value)
        {
            body.Add((byte)(value & 0xFF));
            body.Add((byte)((value >> 8) & 0xFF));
            body.Add((byte)((value >> 16) & 0xFF));
            body.Add((byte)((value >> 24) & 0xFF));
        }

        public static void AddDouble(List<byte> body, double value)
        {
            long bits = BitConverter.DoubleToInt64Bits(value);
            for (int i = 0; i < 8; i++) body.Add((byte)((bits >> (8 * i)) & 0xFF));
        }

        public static void AddString(List<byte> body, string value)
        {
            byte[] encoded = ArchiveText.Utf8Strict.GetBytes(value ?? string.Empty);
            if (encoded.Length > ArchiveText.MaxStringBytes)
                throw new ArgumentException("String exceeds " + ArchiveText.MaxStringBytes +
                                            " UTF-8 bytes.", "value");
            AddInt32(body, encoded.Length);
            body.AddRange(encoded);
        }

        public static string GetString(byte[] body, ref int at)
        {
            int count = GetInt32(body, ref at);
            if (count < 0 || count > ArchiveText.MaxStringBytes)
                throw new FullHandArchiveFormatException(
                    "String length " + count + " is out of range.");
            if (count > body.Length - at)
                throw new FullHandArchiveFormatException("Truncated string body.");
            string value;
            try { value = ArchiveText.Utf8Strict.GetString(body, at, count); }
            catch (DecoderFallbackException error)
            {
                throw new FullHandArchiveFormatException("Invalid UTF-8 in archive.", error);
            }
            at += count;
            return value;
        }

        public static byte[] Sha256(byte[] data)
        {
            using (var hash = SHA256.Create()) { return hash.ComputeHash(data); }
        }

        public static string ToHex(byte[] data)
        {
            var text = new StringBuilder(data.Length * 2);
            for (int i = 0; i < data.Length; i++) text.Append(data[i].ToString("x2"));
            return text.ToString();
        }
    }
}
