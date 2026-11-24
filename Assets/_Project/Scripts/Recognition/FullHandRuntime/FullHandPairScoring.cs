// F4 - fixed logistic pair models and closed-set class scoring. No ML dependency.
//
// Three methods, each reporting its OWN outcome: DTW-only (class minimum cost), ML-only
// (15 fixed-time descriptors) and hybrid (18 aligned descriptors). One method's unusable
// pair never erases another's. There is no ACCEPT, no threshold, no enrollment, no game
// authority, and the expected target is NOT an input to any of this.
using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    public static class FullHandMethods
    {
        public const string DtwOnly = "dtw_only";
        public const string MlOnly = "ml_only";
        public const string Hybrid = "hybrid";
        public static readonly string[] All = { DtwOnly, MlOnly, Hybrid };
    }

    /// <summary>Immutable fitted logistic parameters. Validated on construction.</summary>
    public sealed class FullHandModel
    {
        readonly double[] mean;
        readonly double[] scale;
        readonly double[] coefficients;
        readonly double[] imputation;              // phase models only; null for 15/18

        public int FeatureCount { get; private set; }
        public double Intercept { get; private set; }

        /// <summary>True for a 432-wide monotone phase model (research candidate).</summary>
        public bool IsPhase { get { return imputation != null; } }

        public double[] CopyImputationMean()
        {
            return imputation == null ? null : (double[])imputation.Clone();
        }

        public FullHandModel(double[] scalerMean, double[] scalerScale, double[] coefficients,
                             double intercept, double[] imputationMean = null)
        {
            if (scalerMean == null || scalerScale == null || coefficients == null)
                throw new ArgumentNullException("scalerMean");
            if (scalerMean.Length != scalerScale.Length ||
                scalerMean.Length != coefficients.Length || scalerMean.Length == 0)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "the model arrays must share one positive width");
            bool phase = scalerMean.Length == FullHandPhasePairFeatures.FeatureCount;
            if (!phase &&
                scalerMean.Length != FullHandPairConstants.FixedFeatureCount &&
                scalerMean.Length != FullHandPairConstants.AlignedFeatureCount)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "a model must be 15, 18 or 432 wide, got " + scalerMean.Length);
            for (int index = 0; index < scalerMean.Length; index++)
            {
                Require(scalerMean[index], "scaler_mean");
                Require(coefficients[index], "coefficients");
                Require(scalerScale[index], "scaler_scale");
                if (!(scalerScale[index] > 0.0))
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "a scale must be strictly positive; index " + index);
            }
            Require(intercept, "intercept");
            if (phase)
            {
                // A phase model must carry its own imputation means and must stay monotone:
                // error columns (stat 0 mean, stat 1 p90) may not raise compatibility, coverage
                // columns (stat 2) may not lower it.
                if (imputationMean == null || imputationMean.Length != scalerMean.Length)
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "a 432-wide model requires an explicit 432-wide imputation mean");
                for (int index = 0; index < imputationMean.Length; index++)
                {
                    Require(imputationMean[index], "imputation_mean");
                    bool coverage = index % 3 == 2;
                    if (!coverage && coefficients[index] > 0.0)
                        throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                            "an error coefficient must be <= 0; index " + index);
                    if (coverage && coefficients[index] < 0.0)
                        throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                            "a coverage coefficient must be >= 0; index " + index);
                }
            }
            else if (imputationMean != null)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "only a 432-wide model takes an imputation mean");
            this.mean = (double[])scalerMean.Clone();
            this.scale = (double[])scalerScale.Clone();
            this.coefficients = (double[])coefficients.Clone();
            this.imputation = phase ? (double[])imputationMean.Clone() : null;
            FeatureCount = scalerMean.Length;
            Intercept = intercept;
        }

        static void Require(double value, string name)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    name + " carries a non-finite value");
        }

        public double[] CopyScalerMean() { return (double[])mean.Clone(); }
        public double[] CopyScalerScale() { return (double[])scale.Clone(); }
        public double[] CopyCoefficients() { return (double[])coefficients.Clone(); }

        /// <summary>
        /// Classic 15/18: a missing descriptor standardizes to 0, i.e. the training mean.
        /// Phase 432: NaN means missing and standardizes the model's own imputation mean;
        /// an infinite input is a technical failure, matching the offline checker.
        /// </summary>
        public double[] Standardize(double[] features)
        {
            if (features == null || features.Length != FeatureCount)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "pair feature width does not match the model");
            var standardized = new double[FeatureCount];
            for (int index = 0; index < FeatureCount; index++)
            {
                if (imputation != null)
                {
                    double raw = features[index];
                    if (double.IsInfinity(raw))
                        throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                            "a phase feature must not be infinite; index " + index);
                    standardized[index] = double.IsNaN(raw)
                        ? (imputation[index] - mean[index]) / scale[index]
                        : (raw - mean[index]) / scale[index];
                    continue;
                }
                double value = (features[index] - mean[index]) / scale[index];
                standardized[index] = (double.IsNaN(value) || double.IsInfinity(value))
                    ? 0.0 : value;
            }
            return standardized;
        }

        public double Compatibility(double[] features)
        {
            double[] standardized = Standardize(features);
            double logit = 0.0;
            for (int index = 0; index < FeatureCount; index++)
                logit += standardized[index] * coefficients[index];
            logit += Intercept;
            if (double.IsNaN(logit) || double.IsInfinity(logit))
                throw new FullHandFormatException(FullHandFailureCodes.NonFiniteLogit,
                    "a non-finite logit is a technical failure");
            if (logit >= 0.0) return 1.0 / (1.0 + Math.Exp(-logit));
            double exponential = Math.Exp(logit);
            return exponential / (1.0 + exponential);
        }
    }

    public sealed class FullHandMethodResult
    {
        public string Method;
        public string Status;
        public string Reason;
        public string Top1;
        public double Score = double.NaN;
        public double Margin = double.NaN;
        public int FailedPairs;
        public string[] Classes;
        public double[] RawScores;                  // NaN where the class had no evidence
        public string[] ClassesWithoutEvidence;
    }

    public sealed class FullHandEvaluation
    {
        public string[] Vocabulary;
        public int ReferenceCount;
        public bool QueryUsable;
        public string QueryFailure;
        public FullHandMethodResult[] Methods;

        public FullHandMethodResult Method(string name)
        {
            for (int index = 0; index < Methods.Length; index++)
                if (string.Equals(Methods[index].Method, name, StringComparison.Ordinal))
                    return Methods[index];
            return null;
        }
    }

    /// <summary>
    /// The exact surface FullHandScorer already reads from a bank. Extracting it changes no
    /// math and no format: FullHandRuntimeBundle implements every member as it stands.
    /// </summary>
    public interface IFullHandScoringBank
    {
        string[] CopyVocabulary();
        int ReferenceCount { get; }
        int[] ReferencesForClass(int labelIndex);
        FullHandFeatureSequence ReferenceSequence(int index);
        FullHandModel MlModel { get; }
        FullHandModel HybridModel { get; }
    }

    public static class FullHandScorer
    {
        /// <summary>The unchanged public entry point; it delegates to the bank overload.</summary>
        public static FullHandEvaluation Evaluate(FullHandRuntimeBundle bundle,
                                                  FullHandFeatureSequence query)
        {
            if (bundle == null) throw new ArgumentNullException("bundle");
            return Evaluate((IFullHandScoringBank)bundle, query);
        }

        /// <summary>
        /// Scores EVERY configured class through EVERY allowed reference. DTW takes the
        /// class minimum cost; ML and hybrid take the class maximum compatibility. An exact
        /// tie is broken by ordinal label order.
        /// </summary>
        public static FullHandEvaluation Evaluate(IFullHandScoringBank bundle,
                                                  FullHandFeatureSequence query)
        {
            if (bundle == null) throw new ArgumentNullException("bundle");
            if (query == null) throw new ArgumentNullException("query");
            // A bank is entirely classic or entirely phase; a mixed pair of widths would mean
            // two different feature definitions in one bank and is refused outright.
            bool mlPhase = bundle.MlModel != null && bundle.MlModel.IsPhase;
            bool hybridPhase = bundle.HybridModel != null && bundle.HybridModel.IsPhase;
            if (mlPhase != hybridPhase)
                throw new FullHandFormatException(FullHandFailureCodes.SchemaMismatch,
                    "classic and phase models cannot be mixed in one bank");
            bool phase = mlPhase && hybridPhase;
            string[] vocabulary = bundle.CopyVocabulary();
            var evaluation = new FullHandEvaluation
            {
                Vocabulary = vocabulary,
                ReferenceCount = bundle.ReferenceCount,
                QueryUsable = query.Usable,
                QueryFailure = query.Failure,
                Methods = new FullHandMethodResult[FullHandMethods.All.Length]
            };
            if (!query.Usable)
            {
                // an unavailable SEQUENCE is not a per-method pair failure; it is reported
                // with its own reason for every method
                for (int index = 0; index < FullHandMethods.All.Length; index++)
                    evaluation.Methods[index] = new FullHandMethodResult
                    {
                        Method = FullHandMethods.All[index],
                        Status = FullHandStatus.TechnicalFailure,
                        Reason = query.Failure,
                        Classes = vocabulary,
                        RawScores = FilledWithNaN(vocabulary.Length),
                        ClassesWithoutEvidence = (string[])vocabulary.Clone()
                    };
                return evaluation;
            }

            int classCount = vocabulary.Length;
            var dtw = FilledWithNaN(classCount);
            var ml = FilledWithNaN(classCount);
            var hybrid = FilledWithNaN(classCount);
            var failed = new int[3];                       // dtw, ml, hybrid
            var absent = new List<string>();
            var missing = new List<string>[] { new List<string>(), new List<string>(),
                                               new List<string>() };

            for (int label = 0; label < classCount; label++)
            {
                int[] references = bundle.ReferencesForClass(label);
                if (references.Length == 0) absent.Add(vocabulary[label]);
                bool anyDtw = false, anyMl = false, anyHybrid = false;
                for (int index = 0; index < references.Length; index++)
                {
                    FullHandFeatureSequence reference = bundle.ReferenceSequence(references[index]);
                    FullHandPairResult aligned = phase
                        ? FullHandPhasePairFeatures.AlignedPairFeatures(query, reference)
                        : FullHandPairGeometry.AlignedPairFeatures(query, reference);
                    if (aligned.Ok)
                    {
                        double value = bundle.HybridModel.Compatibility(aligned.FeaturesReference);
                        hybrid[label] = anyHybrid ? Math.Max(hybrid[label], value) : value;
                        double cost = aligned.DtwCost;
                        dtw[label] = anyDtw ? Math.Min(dtw[label], cost) : cost;
                        anyHybrid = true; anyDtw = true;
                    }
                    else { failed[0]++; failed[2]++; }
                    FullHandPairResult fixedTime = phase
                        ? FullHandPhasePairFeatures.FixedTimeFeatures(query, reference)
                        : FullHandPairGeometry.FixedTimeFeatures(query, reference);
                    if (fixedTime.Ok)
                    {
                        double value = bundle.MlModel.Compatibility(fixedTime.FeaturesReference);
                        ml[label] = anyMl ? Math.Max(ml[label], value) : value;
                        anyMl = true;
                    }
                    else failed[1]++;
                }
                if (!anyDtw) missing[0].Add(vocabulary[label]);
                if (!anyMl) missing[1].Add(vocabulary[label]);
                if (!anyHybrid) missing[2].Add(vocabulary[label]);
            }

            var scores = new[] { dtw, ml, hybrid };
            var higher = new[] { false, true, true };
            for (int index = 0; index < FullHandMethods.All.Length; index++)
            {
                var blocked = new List<string>(missing[index]);
                for (int at = 0; at < absent.Count; at++)
                    if (!blocked.Contains(absent[at])) blocked.Add(absent[at]);
                blocked.Sort(StringComparer.Ordinal);
                var result = new FullHandMethodResult
                {
                    Method = FullHandMethods.All[index],
                    Classes = vocabulary,
                    RawScores = (double[])scores[index].Clone(),
                    FailedPairs = failed[index],
                    ClassesWithoutEvidence = blocked.ToArray()
                };
                if (blocked.Count > 0)
                {
                    result.Status = FullHandStatus.TechnicalFailure;
                    result.Reason = absent.Count > 0 ? FullHandStatus.NoReferenceForClass
                                                     : FullHandStatus.NoUsableReferencePair;
                }
                else
                {
                    result.Status = FullHandStatus.Ranked;
                    Rank(vocabulary, scores[index], higher[index], result);
                }
                evaluation.Methods[index] = result;
            }
            return evaluation;
        }

        /// <summary>Best then runner-up in the same ordering the verified Python uses.</summary>
        static void Rank(string[] vocabulary, double[] values, bool higherIsBetter,
                         FullHandMethodResult result)
        {
            var order = new int[vocabulary.Length];
            for (int index = 0; index < order.Length; index++) order[index] = index;
            Array.Sort(order, delegate (int left, int right)
            {
                double a = higherIsBetter ? -values[left] : values[left];
                double b = higherIsBetter ? -values[right] : values[right];
                if (a < b) return -1;
                if (a > b) return 1;
                return string.CompareOrdinal(vocabulary[left], vocabulary[right]);
            });
            result.Top1 = vocabulary[order[0]];
            result.Score = values[order[0]];
            if (order.Length < 2) { result.Margin = double.NaN; return; }
            double runner = values[order[1]];
            result.Margin = higherIsBetter ? result.Score - runner : runner - result.Score;
        }

        static double[] FilledWithNaN(int count)
        {
            var values = new double[count];
            for (int index = 0; index < count; index++) values[index] = double.NaN;
            return values;
        }
    }
}
