// F12 - the minimal opt-in bridge from the finalized full-hand observation to the F11
// review-only candidate archive. Additional API only.
//
// Scope and non-scope:
//   * This file adds NO recognition, NO model fitting, NO threshold, NO enrollment and NO
//     promotion. It moves an already-produced capture and an already-produced evaluation
//     into immutable review storage, and nothing else.
//   * Every record it builds is ReviewOnly with IsEnrollmentEligible false and quality
//     Unassessed. No sample built here is ever training data.
//   * declaredTargetWord is caller metadata only. Nothing here compares it to the
//     predicted word, and no branch anywhere depends on a word name.
//   * There is no default root: a file sink requires one explicitly supplied, existing,
//     fully qualified directory outside Assets/StreamingAssets. Nothing here reads the
//     environment, and an existing archive file is never replaced or retried over.
//   * The codec, the envelope and the on-disk layout are entirely F11's; this file
//     recreates none of them.
using System;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Personalization;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    /// <summary>
    /// The typed archive seam. One finished review-only record in, storage out. An
    /// implementation may throw: the observer treats any throw as an archive failure and
    /// keeps its own result, status and counters untouched.
    /// </summary>
    public interface IFullHandReviewCandidateSink
    {
        void Write(FullHandReviewCandidate candidate);
    }

    /// <summary>
    /// Saves each record under one explicit, already existing absolute directory through
    /// the F11 store, which never overwrites an existing file.
    /// </summary>
    public sealed class FullHandFileReviewCandidateSink : IFullHandReviewCandidateSink
    {
        readonly string root;

        public FullHandFileReviewCandidateSink(string root)
        {
            if (string.IsNullOrEmpty(root))
                throw new FullHandFormatException("ARCHIVE_ROOT_MISSING", "no root");
            // checked on the RAW value: GetFullPath would silently complete a relative,
            // drive-relative or root-relative path against the current directory
            if (!FullHandFileObservationSink.IsFullyQualified(root))
                throw new FullHandFormatException("ARCHIVE_ROOT_NOT_ABSOLUTE",
                    "a fully qualified directory is required");
            string full = System.IO.Path.GetFullPath(root);
            if (!System.IO.Directory.Exists(full))
                throw new FullHandFormatException("ARCHIVE_ROOT_MISSING",
                    "the archive root must already exist");
            foreach (string segment in full.Split(System.IO.Path.DirectorySeparatorChar,
                                                  System.IO.Path.AltDirectorySeparatorChar))
                if (string.Equals(segment, "Assets", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(segment, "StreamingAssets", StringComparison.OrdinalIgnoreCase))
                    throw new FullHandFormatException("ARCHIVE_ROOT_REFUSED",
                        "an archive root under Assets or StreamingAssets is refused");
            this.root = full;
        }

        /// <summary>The normalized root actually in use.</summary>
        public string Root { get { return root; } }

        /// <summary>Delegates to F11 storage; an existing file is never replaced.</summary>
        public void Write(FullHandReviewCandidate candidate)
        {
            FullHandReviewCandidateStore.Save(candidate, root);
        }
    }

    /// <summary>
    /// Builds the review-only record from what the observation already computed. It runs no
    /// classifier, renormalizes nothing, and supplies no target to any scorer.
    /// </summary>
    public static class FullHandReviewArchiveBuilder
    {
        /// <summary>The existing per-session record number, e.g. "observation_0001".</summary>
        public static string SampleId(int index)
        {
            return "observation_" + index.ToString("0000");
        }

        /// <summary>
        /// Word evidence taken from the EXISTING hybrid entry of the existing evaluation:
        /// scored only when that one method reported Ranked with a usable finite score,
        /// unscored otherwise. No method is chosen per word and no method is re-run.
        /// </summary>
        public static FullHandWordEvidence WordEvidenceFrom(FullHandEvaluation evaluation,
                                                            string modelIdentitySha256Hex)
        {
            if (evaluation == null) return FullHandWordEvidence.Unscored();
            FullHandMethodResult hybrid = evaluation.Method(FullHandMethods.Hybrid);
            if (hybrid == null ||
                !string.Equals(hybrid.Status, FullHandStatus.Ranked, StringComparison.Ordinal))
                return FullHandWordEvidence.Unscored();
            if (string.IsNullOrEmpty(hybrid.Top1) ||
                double.IsNaN(hybrid.Score) || double.IsInfinity(hybrid.Score))
                return FullHandWordEvidence.Unscored();
            return FullHandWordEvidence.Scored(hybrid.Top1, hybrid.Score,
                                               modelIdentitySha256Hex,
                                               FullHandPairConstants.PairSchemaId);
        }

        /// <summary>
        /// The whole record. Quality is always Unassessed here: this bridge holds no
        /// assessor authority, and the archive keeps IsEnrollmentEligible false regardless.
        /// Any invalid input (an empty declared target above all) throws out of here, which
        /// the caller records as a typed archive failure rather than inventing a label.
        /// </summary>
        public static FullHandReviewCandidate Build(string sampleId, string declaredTarget,
                                                    FullHandCommonCapture capture,
                                                    double[] timestamps,
                                                    FullHandEvaluation evaluation,
                                                    string modelIdentitySha256Hex)
        {
            return new FullHandReviewCandidate(
                capture, timestamps, sampleId, declaredTarget,
                WordEvidenceFrom(evaluation, modelIdentitySha256Hex),
                FullHandQualityEvidence.Unassessed());
        }
    }
}
