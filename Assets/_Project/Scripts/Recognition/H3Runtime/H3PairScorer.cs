// M3_H3_RUNTIME_R1 - faithful C# port of the FROZEN H3 v1 shared pair model and gates.
// Prepared in the workspace. NOT installed and NOT wired to any recognizer.
//
// Ported from, and kept identical to:
//   h3_pair_model.PairModel / stable_sigmoid / validate_raw_scores / score_classes /
//   decide_h3, and hybrid_core_v2.decide for the relative accept policy.
//
// ONE shared class-agnostic model answers same-word or different-word for a PAIR. There
// is no per-word model, no per-word case, no confidence-only fallback and no rule by
// which DTW overrides the model. A class is scored by the MAXIMUM same-word compatibility
// over its K references, and every reference of every configured class is evaluated.
//
// The relative share is scores / sum(scores). It is a reporting convenience so the
// existing accept 0.60 / margin 0.15 policy applies unchanged. It is NOT a calibrated
// multiclass posterior: nothing was fit to produce one, and the denominator depends on
// how many classes are configured.
using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.H3Runtime
{
    public static class H3ScoringCodes
    {
        public const string PairFeatureWidthMismatch = "PAIR_FEATURE_WIDTH_MISMATCH";
        public const string NonfinitePairFeature = "NONFINITE_PAIR_FEATURE";
        public const string NonfiniteTransform = "NONFINITE_TRANSFORM";
        public const string NonfiniteLogit = "NONFINITE_LOGIT";

        public const string NoValidReferencePair = "NO_VALID_REFERENCE_PAIR";
        public const string MalformedScoreVocabulary = "MALFORMED_SCORE_VOCABULARY";
        public const string MalformedCompatibilityValue = "MALFORMED_COMPATIBILITY_VALUE";
        public const string NonfiniteCompatibility = "NONFINITE_COMPATIBILITY";
        public const string CompatibilityOutOfRange = "COMPATIBILITY_OUT_OF_RANGE";
        public const string AllCompatibilityZero = "ALL_COMPATIBILITY_ZERO";
        public const string InvalidProbabilities = "INVALID_PROBABILITIES";
    }

    public static class H3Status
    {
        public const string Accept = "ACCEPT";
        public const string Abstain = "ABSTAIN";
        public const string TechnicalFailure = "TECHNICAL_FAILURE";
    }

    public static class H3Decision
    {
        // frozen, and the bundle loader refuses a bundle that declares anything else
        public const double RelativeAcceptProbability = 0.60;
        public const double RelativeAcceptMargin = 0.15;
        public const double AbsoluteCompatibilityFloor = 0.50;

        public const double CompatibilityMinimum = 0.0;
        public const double CompatibilityMaximum = 1.0;
    }

    /// <summary>The shared pair model. Pure arithmetic over the exported numbers.</summary>
    public sealed class H3PairModel
    {
        readonly double[] mean;
        readonly double[] scale;
        readonly double[] coefficients;
        readonly double intercept;

        public H3PairModel(double[] scalerMean, double[] scalerScale,
                           double[] coefficients, double intercept)
        {
            this.mean = scalerMean; this.scale = scalerScale;
            this.coefficients = coefficients; this.intercept = intercept;
        }

        public static H3PairModel FromBundle(H3RuntimeBundle bundle)
        {
            return new H3PairModel(bundle.ScalerMean, bundle.ScalerScale,
                                   bundle.Coefficients, bundle.Intercept);
        }

        /// <summary>
        /// Saturating logistic. A large magnitude logit is a confident answer, not an
        /// overflow: both branches underflow to 0.0 instead, so the result stays in [0, 1]
        /// for every finite input.
        /// </summary>
        public static double StableSigmoid(double z)
        {
            if (z >= 0.0) return 1.0 / (1.0 + Math.Exp(-z));
            double exponential = Math.Exp(z);
            return exponential / (1.0 + exponential);
        }

        public double SameWordProbability(double[] features)
        {
            if (features == null || features.Length != coefficients.Length)
                throw new H3FormatException(H3ScoringCodes.PairFeatureWidthMismatch,
                    "pair feature width does not match the model");
            double logit = intercept;
            for (int index = 0; index < features.Length; index++)
            {
                double value = features[index];
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new H3FormatException(H3ScoringCodes.NonfinitePairFeature,
                        "a nonfinite pair feature is a technical failure and must never "
                        + "be substituted");
                double standardized = (value - mean[index]) / scale[index];
                if (double.IsNaN(standardized) || double.IsInfinity(standardized))
                    throw new H3FormatException(H3ScoringCodes.NonfiniteTransform,
                        "the standardizing transform produced a nonfinite value");
                logit += standardized * coefficients[index];
            }
            if (double.IsNaN(logit) || double.IsInfinity(logit))
                throw new H3FormatException(H3ScoringCodes.NonfiniteLogit,
                    "the computed logit is not finite");
            return StableSigmoid(logit);
        }
    }

    public sealed class H3ClassScores
    {
        public bool Ok { get; private set; }
        public string Failure { get; private set; }
        public string[] Classes { get; private set; }
        public double[] Raw { get; private set; }
        public double[][] PerReference { get; private set; }
        public int[] FailedReferences { get; private set; }

        public H3ClassScores(bool ok, string failure, string[] classes, double[] raw,
                             double[][] perReference, int[] failedReferences)
        {
            Ok = ok; Failure = failure; Classes = classes; Raw = raw;
            PerReference = perReference; FailedReferences = failedReferences;
        }
    }

    public sealed class H3DecisionResult
    {
        public string Status;
        public string QualityCode;
        public string Top1;
        public string Predicted;
        public string[] Classes;
        public double[] RawCompatibility;
        public double[] RelativeShareNotAPosterior;
        public double BestRawCompatibility;
        public double RunnerUpRawCompatibility;
        public bool? RelativeGatePassed;
        public bool? AbsoluteGatePassed;
        public int[] FailedReferences;
        public string Note = string.Empty;
    }

    public static class H3Scorer
    {
        /// <summary>
        /// Scores every configured class by the MAXIMUM same-word compatibility over its
        /// references. A reference whose pair failed is counted, never substituted, and a
        /// class with no surviving reference makes the whole input a technical failure,
        /// because it cannot be ranked against the others on equal terms.
        /// </summary>
        public static H3ClassScores ScoreClasses(H3RuntimeBundle bundle, H3PairModel model,
                                                 double[] queryCoords, bool[] queryMask,
                                                 int queryFrames)
        {
            string[] classes = bundle.Vocabulary;
            var raw = new double[classes.Length];
            var perReference = new double[classes.Length][];
            var failed = new int[classes.Length];
            for (int index = 0; index < classes.Length; index++)
            {
                H3ReferenceSequence[] references = bundle.Bank[classes[index]];
                var scores = new List<double>(references.Length);
                for (int reference = 0; reference < references.Length; reference++)
                {
                    H3ReferenceSequence entry = references[reference];
                    PairFeatureResult pair = H3PairGeometry.PairFeatures(
                        queryCoords, queryMask, queryFrames,
                        entry.Coordinates, entry.Mask, entry.FrameCount);
                    if (!pair.Ok) { failed[index]++; continue; }
                    scores.Add(model.SameWordProbability(pair.Features));
                }
                perReference[index] = scores.ToArray();
                if (scores.Count == 0)
                    return new H3ClassScores(false, H3ScoringCodes.NoValidReferencePair,
                                             classes, null, perReference, failed);
                double best = scores[0];
                for (int at = 1; at < scores.Count; at++)
                    if (scores[at] > best) best = scores[at];
                raw[index] = best;
            }
            return new H3ClassScores(true, null, classes, raw, perReference, failed);
        }

        /// <summary>
        /// Validates the raw scores BEFORE anything is normalized. Normalizing first turns
        /// a malformed or out-of-range vector into a confident looking share.
        /// </summary>
        public static string ValidateRawScores(string[] classes, double[] raw)
        {
            if (classes == null || raw == null || classes.Length == 0)
                return H3ScoringCodes.MalformedScoreVocabulary;
            if (classes.Length != raw.Length)
                return H3ScoringCodes.MalformedScoreVocabulary;
            var seen = new Dictionary<string, bool>(StringComparer.Ordinal);
            for (int index = 0; index < classes.Length; index++)
            {
                if (string.IsNullOrEmpty(classes[index])
                    || seen.ContainsKey(classes[index]))
                    return H3ScoringCodes.MalformedScoreVocabulary;
                seen[classes[index]] = true;
            }
            for (int index = 0; index < raw.Length; index++)
            {
                if (double.IsNaN(raw[index]) || double.IsInfinity(raw[index]))
                    return H3ScoringCodes.NonfiniteCompatibility;
                if (raw[index] < H3Decision.CompatibilityMinimum
                    || raw[index] > H3Decision.CompatibilityMaximum)
                    return H3ScoringCodes.CompatibilityOutOfRange;
            }
            return null;
        }

        /// <summary>The unchanged relative accept policy, applied to the share vector.</summary>
        static bool RelativeGate(double[] share, out int top1Index, out string failure)
        {
            failure = null;
            top1Index = -1;
            double sum = 0.0;
            for (int index = 0; index < share.Length; index++)
            {
                if (double.IsNaN(share[index]) || double.IsInfinity(share[index])
                    || share[index] < -1e-9)
                {
                    failure = H3ScoringCodes.InvalidProbabilities;
                    return false;
                }
                sum += share[index];
            }
            double tolerance = Math.Max(1e-9 * Math.Max(Math.Abs(sum), 1.0), 1e-6);
            if (Math.Abs(sum - 1.0) > tolerance)
            {
                failure = H3ScoringCodes.InvalidProbabilities;
                return false;
            }
            // first maximum in class order wins a tie, matching the Python dictionary max
            top1Index = 0;
            for (int index = 1; index < share.Length; index++)
                if (share[index] > share[top1Index]) top1Index = index;

            var ordered = (double[])share.Clone();
            Array.Sort(ordered);
            Array.Reverse(ordered);
            double top = ordered[0];
            double margin = ordered.Length > 1 ? top - ordered[1] : top;
            return top >= H3Decision.RelativeAcceptProbability
                   && margin >= H3Decision.RelativeAcceptMargin;
        }

        /// <summary>
        /// Ranks by raw compatibility, then requires BOTH gates: the unchanged relative
        /// policy and an absolute floor on the best raw compatibility. The absolute floor
        /// exists because a relative share can be high while every reference looks unlike
        /// the query, and even that is not unknown-gesture evidence.
        /// </summary>
        public static H3DecisionResult Decide(string qualityCode, H3ClassScores scores,
                                              string[] classes)
        {
            var result = new H3DecisionResult
            {
                Classes = classes,
                FailedReferences = scores == null ? null : scores.FailedReferences,
                QualityCode = qualityCode
            };
            if (qualityCode != H3FailureCodes.Ok)
            {
                result.Status = H3Status.TechnicalFailure;
                result.Note = "rejected before the model was consulted";
                return result;
            }
            if (scores == null || !scores.Ok)
            {
                result.Status = H3Status.TechnicalFailure;
                result.QualityCode = scores == null ? "NO_SCORES" : scores.Failure;
                result.Note = "no class could be scored on equal terms";
                return result;
            }
            string problem = ValidateRawScores(classes, scores.Raw);
            if (problem != null)
            {
                result.Status = H3Status.TechnicalFailure;
                result.QualityCode = problem;
                result.Note = "the raw compatibility vector failed its contract; "
                              + "normalizing it would manufacture a confident share";
                return result;
            }

            double total = 0.0;
            for (int index = 0; index < scores.Raw.Length; index++) total += scores.Raw[index];
            result.RawCompatibility = (double[])scores.Raw.Clone();
            if (!(total > 0.0))
            {
                result.Status = H3Status.TechnicalFailure;
                result.QualityCode = H3ScoringCodes.AllCompatibilityZero;
                result.Note = "no reference was compatible enough to rank";
                return result;
            }

            var share = new double[scores.Raw.Length];
            for (int index = 0; index < share.Length; index++)
                share[index] = scores.Raw[index] / total;
            var ordered = (double[])scores.Raw.Clone();
            Array.Sort(ordered);
            Array.Reverse(ordered);
            result.RelativeShareNotAPosterior = share;
            result.BestRawCompatibility = ordered[0];
            result.RunnerUpRawCompatibility = ordered.Length > 1 ? ordered[1] : 0.0;

            int top1Index;
            string relativeFailure;
            bool relativePassed = RelativeGate(share, out top1Index, out relativeFailure);
            if (relativeFailure != null)
            {
                result.Status = H3Status.TechnicalFailure;
                result.QualityCode = relativeFailure;
                result.Note = "the relative share vector failed validation";
                return result;
            }
            result.Top1 = classes[top1Index];
            bool absolutePassed =
                result.BestRawCompatibility >= H3Decision.AbsoluteCompatibilityFloor;
            result.RelativeGatePassed = relativePassed;
            result.AbsoluteGatePassed = absolutePassed;

            if (relativePassed && absolutePassed)
            {
                result.Status = H3Status.Accept;
                result.Predicted = result.Top1;
                result.Note = "a high compatibility is not evidence that an unknown or "
                              + "incorrect gesture would be rejected";
                return result;
            }
            result.Status = H3Status.Abstain;
            result.Note = !relativePassed
                ? "below the relative accept policy"
                : "best raw compatibility below the absolute floor";
            return result;
        }

        /// <summary>Quality, resample, score, decide. The whole frozen inference path.</summary>
        public static H3DecisionResult Evaluate(H3RuntimeBundle bundle, H3PairModel model,
                                                double[] rawCoords, bool[] rawMask,
                                                int frameCount)
        {
            QualityResult quality =
                H3PairGeometry.CheckInputQuality(rawCoords, rawMask, frameCount);
            if (!quality.Ok) return Decide(quality.Code, null, bundle.Vocabulary);
            double[] resampledCoords;
            bool[] resampledMask;
            H3PairGeometry.ResampleToFixed(rawCoords, rawMask, frameCount,
                                           out resampledCoords, out resampledMask);
            H3ClassScores scores = ScoreClasses(bundle, model, resampledCoords,
                                                resampledMask, H3Constants.ResampleFrames);
            return Decide(H3FailureCodes.Ok, scores, bundle.Vocabulary);
        }
    }
}
