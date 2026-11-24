// F20 - shared 46-joint recognition pipeline. Infrastructure only.
//
// This file adds NO classifier, NO new geometry and NO fitted number. Every score comes
// from the existing F4 math through FullHandScorer; the gates only reproduce the frozen
// Python accept/abstain contract with thresholds the CALLER supplies. There is no file
// I/O, no environment lookup, no profile access, no camera, no scheduler and no game
// success authority here: a gate "Accepted" is a word-level bank decision, not a
// verification that the sign was performed well.
//
// The current-participant addition bank is RESEARCH ONLY. Earlier measurements showed
// that adding personal references can raise false accepts, so nothing here activates,
// persists or enrolls anything, and no improvement is claimed.
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    public static class FullHandDecisions
    {
        public const string Accepted = "ACCEPTED";
        public const string Abstained = "ABSTAINED";
        public const string TechnicalFailure = "TECHNICAL_FAILURE";
        public const string UnsupportedTarget = "UNSUPPORTED_TARGET";
    }

    public static class FullHandPipelineBanks
    {
        public const string CommonOnly = "common_only";
        public const string CommonPlusParticipant = "common_plus_participant";
    }

    public static class FullHandPipelineCodes
    {
        public const string IdentifierInvalid = "PIPELINE_IDENTIFIER_INVALID";
        public const string HashShapeInvalid = "PIPELINE_HASH_SHAPE_INVALID";
        public const string ParticipantMismatch = "PIPELINE_PARTICIPANT_MISMATCH";
        public const string LabelUnsupported = "PIPELINE_LABEL_UNSUPPORTED";
        public const string DuplicateSample = "PIPELINE_DUPLICATE_SAMPLE";
        public const string GateInvalid = "PIPELINE_GATE_INVALID";
        public const string GateSetIncomplete = "PIPELINE_GATE_SET_INCOMPLETE";
        public const string BankInvalid = "PIPELINE_BANK_INVALID";
        public const string UnknownMethod = "PIPELINE_UNKNOWN_METHOD";
    }

    // ---- banks ---------------------------------------------------------------

    /// <summary>
    /// A common bank the pipeline may read: the scoring surface plus the opaque reference
    /// ids and features it needs for self/alias exclusion. Source hashes are NOT part of a
    /// common bundle and are never invented here.
    /// </summary>
    public interface IFullHandCommonBank : IFullHandScoringBank
    {
        string ModelId { get; }
        string[] CopyReferenceIds();
    }

    /// <summary>Adapts a loaded bundle. The bundle is read, never copied or rewritten.</summary>
    public sealed class FullHandBundleCommonBank : IFullHandCommonBank
    {
        readonly FullHandRuntimeBundle bundle;

        public FullHandBundleCommonBank(FullHandRuntimeBundle bundle)
        {
            if (bundle == null) throw new ArgumentNullException("bundle");
            this.bundle = bundle;
        }

        public string ModelId { get { return bundle.PayloadSha256; } }
        public FullHandModel MlModel { get { return bundle.MlModel; } }
        public FullHandModel HybridModel { get { return bundle.HybridModel; } }
        public int ReferenceCount { get { return bundle.ReferenceCount; } }
        public string[] CopyVocabulary() { return bundle.CopyVocabulary(); }
        public string[] CopyReferenceIds() { return bundle.CopyReferenceIds(); }
        public int[] ReferencesForClass(int labelIndex) { return bundle.ReferencesForClass(labelIndex); }
        public FullHandFeatureSequence ReferenceSequence(int index)
        {
            return bundle.ReferenceSequence(index);
        }
    }

    /// <summary>
    /// An explicit in-memory common bank. Everything is supplied by the caller; nothing is
    /// searched, loaded or defaulted. Useful for tests and for hosting an already-parsed
    /// bank without touching disk.
    /// </summary>
    public sealed class FullHandInMemoryCommonBank : IFullHandCommonBank
    {
        readonly string[] vocabulary;
        readonly string[] referenceIds;
        readonly FullHandFeatureSequence[] sequences;
        readonly int[][] byClass;

        public string ModelId { get; private set; }
        public FullHandModel MlModel { get; private set; }
        public FullHandModel HybridModel { get; private set; }
        public int ReferenceCount { get { return sequences.Length; } }

        public FullHandInMemoryCommonBank(string modelId, string[] vocabulary,
                                          string[] referenceIds, int[] referenceLabelIndices,
                                          FullHandFeatureSequence[] referenceSequences,
                                          FullHandModel mlModel, FullHandModel hybridModel)
        {
            ModelId = FullHandPipelineText.RequireIdentifier(modelId, "modelId");
            if (vocabulary == null || vocabulary.Length == 0)
                throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid, "vocabulary");
            if (referenceIds == null || referenceLabelIndices == null || referenceSequences == null ||
                referenceIds.Length != referenceLabelIndices.Length ||
                referenceIds.Length != referenceSequences.Length)
                throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid,
                    "the reference tables must share one length");
            if (mlModel == null || hybridModel == null)
                throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid, "models");

            this.vocabulary = (string[])vocabulary.Clone();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            for (int index = 0; index < this.vocabulary.Length; index++)
                if (!seen.Add(FullHandPipelineText.RequireIdentifier(this.vocabulary[index], "label")))
                    throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid,
                        "a duplicate label");

            this.referenceIds = (string[])referenceIds.Clone();
            this.sequences = (FullHandFeatureSequence[])referenceSequences.Clone();
            var lists = new List<int>[this.vocabulary.Length];
            for (int index = 0; index < lists.Length; index++) lists[index] = new List<int>();
            seen.Clear();
            for (int index = 0; index < this.referenceIds.Length; index++)
            {
                if (!seen.Add(FullHandPipelineText.RequireIdentifier(this.referenceIds[index],
                                                                     "referenceId")))
                    throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid,
                        "a duplicate reference id");
                if (this.sequences[index] == null)
                    throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid,
                        "a null reference sequence");
                int label = referenceLabelIndices[index];
                if (label < 0 || label >= this.vocabulary.Length)
                    throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid,
                        "unknown label index");
                lists[label].Add(index);
            }
            byClass = new int[this.vocabulary.Length][];
            for (int index = 0; index < lists.Length; index++) byClass[index] = lists[index].ToArray();
            MlModel = mlModel;
            HybridModel = hybridModel;
        }

        public string[] CopyVocabulary() { return (string[])vocabulary.Clone(); }
        public string[] CopyReferenceIds() { return (string[])referenceIds.Clone(); }
        public int[] ReferencesForClass(int labelIndex) { return (int[])byClass[labelIndex].Clone(); }
        public FullHandFeatureSequence ReferenceSequence(int index) { return sequences[index]; }
    }

    // ---- gates ---------------------------------------------------------------

    /// <summary>
    /// One frozen per-method gate. It reproduces the existing Python contract and nothing
    /// more: a gate either rejects everything, or accepts a finite quality-oriented value
    /// q at or above its threshold. q = -score for DTW (a cost) and q = score for the two
    /// logistic methods. q is NOT a probability that the sign was good.
    /// </summary>
    public sealed class FullHandGate
    {
        public string GateId { get; private set; }
        public string Method { get; private set; }
        public bool RejectAll { get; private set; }
        public double QualityThreshold { get; private set; }

        FullHandGate(string gateId, string method, bool rejectAll, double threshold)
        {
            GateId = gateId; Method = method; RejectAll = rejectAll; QualityThreshold = threshold;
        }

        public static FullHandGate RejectEverything(string gateId, string method)
        {
            return new FullHandGate(FullHandPipelineText.RequireIdentifier(gateId, "gateId"),
                                    RequireMethod(method), true, double.NaN);
        }

        public static FullHandGate AtQualityThreshold(string gateId, string method, double quality)
        {
            if (double.IsNaN(quality) || double.IsInfinity(quality))
                throw new FullHandFormatException(FullHandPipelineCodes.GateInvalid,
                    "a threshold must be finite or the gate must reject all");
            return new FullHandGate(FullHandPipelineText.RequireIdentifier(gateId, "gateId"),
                                    RequireMethod(method), false, quality);
        }

        /// <summary>q = -score for DTW, q = score otherwise. No other transform exists.</summary>
        public static double Quality(string method, double score)
        {
            return string.Equals(RequireMethod(method), FullHandMethods.DtwOnly,
                                 StringComparison.Ordinal) ? -score : score;
        }

        /// <summary>Equality accepts. A non-finite quality is never accepted.</summary>
        public bool AcceptsQuality(double quality)
        {
            if (RejectAll) return false;
            if (double.IsNaN(quality) || double.IsInfinity(quality)) return false;
            return quality >= QualityThreshold;
        }

        internal static string RequireMethod(string method)
        {
            for (int index = 0; index < FullHandMethods.All.Length; index++)
                if (string.Equals(FullHandMethods.All[index], method, StringComparison.Ordinal))
                    return method;
            throw new FullHandFormatException(FullHandPipelineCodes.UnknownMethod,
                method ?? "absent");
        }
    }

    /// <summary>Exactly one gate for each of the three methods, under one declared id.</summary>
    public sealed class FullHandGateConfig
    {
        readonly FullHandGate[] gates;
        public string ConfigId { get; private set; }

        public FullHandGateConfig(string configId, FullHandGate[] gates)
        {
            ConfigId = FullHandPipelineText.RequireIdentifier(configId, "configId");
            if (gates == null || gates.Length != FullHandMethods.All.Length)
                throw new FullHandFormatException(FullHandPipelineCodes.GateSetIncomplete,
                    "exactly " + FullHandMethods.All.Length + " gates are required");
            this.gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
            {
                if (gates[index] == null)
                    throw new FullHandFormatException(FullHandPipelineCodes.GateSetIncomplete,
                        "a null gate");
                int slot = MethodIndex(gates[index].Method);
                if (this.gates[slot] != null)
                    throw new FullHandFormatException(FullHandPipelineCodes.GateSetIncomplete,
                        "two gates for " + gates[index].Method);
                this.gates[slot] = gates[index];
            }
        }

        public FullHandGate Gate(string method)
        {
            return gates[MethodIndex(FullHandGate.RequireMethod(method))];
        }

        internal static int MethodIndex(string method)
        {
            for (int index = 0; index < FullHandMethods.All.Length; index++)
                if (string.Equals(FullHandMethods.All[index], method, StringComparison.Ordinal))
                    return index;
            throw new FullHandFormatException(FullHandPipelineCodes.UnknownMethod,
                method ?? "absent");
        }
    }

    // ---- samples -------------------------------------------------------------

    internal static class FullHandPipelineText
    {
        internal const int MaxIdentifier = 128;

        internal static string RequireIdentifier(string value, string name)
        {
            if (string.IsNullOrEmpty(value) || value.Length > MaxIdentifier)
                throw new FullHandFormatException(FullHandPipelineCodes.IdentifierInvalid, name);
            for (int index = 0; index < value.Length; index++)
                if (char.IsControl(value[index]) || char.IsWhiteSpace(value[index]))
                    throw new FullHandFormatException(FullHandPipelineCodes.IdentifierInvalid, name);
            return value;
        }

        /// <summary>Null or empty means UNKNOWN. Anything else must be 64 lowercase-able hex.</summary>
        internal static string NormalizeOptionalSha256(string value, string name)
        {
            if (string.IsNullOrEmpty(value)) return null;
            if (value.Length != 64)
                throw new FullHandFormatException(FullHandPipelineCodes.HashShapeInvalid, name);
            var text = new StringBuilder(64);
            for (int index = 0; index < 64; index++)
            {
                char c = value[index];
                bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
                if (!hex) throw new FullHandFormatException(FullHandPipelineCodes.HashShapeInvalid, name);
                text.Append(char.ToLowerInvariant(c));
            }
            return text.ToString();
        }
    }

    /// <summary>
    /// One immutable sample value. A reference sample carries the class it stands for; a
    /// query sample does not. There is deliberately no "correct" flag: supplying a sample
    /// here is an explicit comparison request, never an approved enrollment and never a
    /// declaration that a sign was performed correctly.
    /// </summary>
    public sealed class FullHandSample
    {
        public string SampleId { get; private set; }
        public string ParticipantId { get; private set; }
        public string SourceSha256 { get; private set; }        // null when unknown
        public string Label { get; private set; }               // null for a query
        public FullHandFeatureSequence Sequence { get; private set; }
        public string FeatureDigest { get; private set; }

        public FullHandSample(string sampleId, string participantId, string sourceSha256,
                              string label, FullHandFeatureSequence sequence)
        {
            SampleId = FullHandPipelineText.RequireIdentifier(sampleId, "sampleId");
            ParticipantId = FullHandPipelineText.RequireIdentifier(participantId, "participantId");
            SourceSha256 = FullHandPipelineText.NormalizeOptionalSha256(sourceSha256, "sourceSha256");
            if (label != null) FullHandPipelineText.RequireIdentifier(label, "label");
            Label = label;
            if (sequence == null) throw new ArgumentNullException("sequence");
            Sequence = sequence;
            FeatureDigest = FullHandFeatureDigest.Compute(sequence);
        }
    }

    /// <summary>Deterministic digest over the immutable points and mask of a sequence.</summary>
    public static class FullHandFeatureDigest
    {
        public static string Compute(FullHandFeatureSequence sequence)
        {
            if (sequence == null) throw new ArgumentNullException("sequence");
            bool[] mask = sequence.CopyMask();
            double[] points = sequence.CopyPoints();
            int dims = FullHandPairConstants.CoordDims;
            var bytes = new byte[mask.Length + CountValid(mask) * dims * 8];
            int at = 0;
            for (int slot = 0; slot < mask.Length; slot++) bytes[at++] = mask[slot] ? (byte)1 : (byte)0;
            // masked slots are skipped entirely, so no NaN bit pattern can reach the digest
            for (int slot = 0; slot < mask.Length; slot++)
            {
                if (!mask[slot]) continue;
                for (int axis = 0; axis < dims; axis++)
                {
                    long bits = BitConverter.DoubleToInt64Bits(points[slot * dims + axis]);
                    for (int shift = 0; shift < 8; shift++)
                        bytes[at++] = (byte)((bits >> (shift * 8)) & 0xFF);
                }
            }
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(bytes);
                var text = new StringBuilder(digest.Length * 2);
                for (int index = 0; index < digest.Length; index++)
                    text.Append(digest[index].ToString("x2"));
                return text.ToString();
            }
        }

        static int CountValid(bool[] mask)
        {
            int total = 0;
            for (int index = 0; index < mask.Length; index++) if (mask[index]) total++;
            return total;
        }
    }

    /// <summary>
    /// The explicit, copied, immutable reference set of ONE participant. Duplicate ids,
    /// duplicate known source hashes and duplicate feature digests are refused, and a
    /// sample belonging to somebody else is refused.
    /// </summary>
    public sealed class FullHandParticipantReferenceSet
    {
        readonly FullHandSample[] samples;
        public string ParticipantId { get; private set; }
        public int Count { get { return samples.Length; } }

        public FullHandParticipantReferenceSet(string participantId, FullHandSample[] samples)
        {
            ParticipantId = FullHandPipelineText.RequireIdentifier(participantId, "participantId");
            var copy = samples == null ? new FullHandSample[0] : (FullHandSample[])samples.Clone();
            var ids = new HashSet<string>(StringComparer.Ordinal);
            var hashes = new HashSet<string>(StringComparer.Ordinal);
            var digests = new HashSet<string>(StringComparer.Ordinal);
            for (int index = 0; index < copy.Length; index++)
            {
                FullHandSample sample = copy[index];
                if (sample == null)
                    throw new FullHandFormatException(FullHandPipelineCodes.IdentifierInvalid,
                        "a null reference sample");
                if (!string.Equals(sample.ParticipantId, ParticipantId, StringComparison.Ordinal))
                    throw new FullHandFormatException(FullHandPipelineCodes.ParticipantMismatch,
                        sample.SampleId);
                if (sample.Label == null)
                    throw new FullHandFormatException(FullHandPipelineCodes.LabelUnsupported,
                        "a reference sample must declare its class");
                if (!ids.Add(sample.SampleId))
                    throw new FullHandFormatException(FullHandPipelineCodes.DuplicateSample,
                        "id " + sample.SampleId);
                if (sample.SourceSha256 != null && !hashes.Add(sample.SourceSha256))
                    throw new FullHandFormatException(FullHandPipelineCodes.DuplicateSample,
                        "source hash of " + sample.SampleId);
                if (!digests.Add(sample.FeatureDigest))
                    throw new FullHandFormatException(FullHandPipelineCodes.DuplicateSample,
                        "feature digest of " + sample.SampleId);
            }
            this.samples = copy;
        }

        public static FullHandParticipantReferenceSet Empty(string participantId)
        {
            return new FullHandParticipantReferenceSet(participantId, null);
        }

        public FullHandSample At(int index) { return samples[index]; }
        public FullHandSample[] CopySamples() { return (FullHandSample[])samples.Clone(); }
    }

    // ---- results -------------------------------------------------------------

    public sealed class FullHandMethodDecision
    {
        readonly string[] classes;
        readonly double[] rawScores;
        readonly string[] classesWithoutEvidence;

        public string Method { get; private set; }
        public string Status { get; private set; }
        public string Decision { get; private set; }
        public string Reason { get; private set; }
        public string Top1 { get; private set; }
        public double Score { get; private set; }
        public double Margin { get; private set; }
        public double Quality { get; private set; }
        public int FailedPairs { get; private set; }
        public string GateId { get; private set; }
        public bool TargetSupported { get; private set; }
        public bool TargetMatches { get; private set; }

        internal FullHandMethodDecision(FullHandMethodResult result, FullHandGate gate,
                                        string declaredTarget, bool targetSupported)
        {
            Method = result.Method;
            Status = result.Status;
            Reason = result.Reason;
            Top1 = result.Top1;
            Score = result.Score;
            Margin = result.Margin;
            FailedPairs = result.FailedPairs;
            GateId = gate.GateId;
            classes = result.Classes == null ? new string[0] : (string[])result.Classes.Clone();
            rawScores = result.RawScores == null ? new double[0] : (double[])result.RawScores.Clone();
            classesWithoutEvidence = result.ClassesWithoutEvidence == null
                ? new string[0] : (string[])result.ClassesWithoutEvidence.Clone();

            bool ranked = string.Equals(result.Status, FullHandStatus.Ranked, StringComparison.Ordinal);
            Quality = ranked ? FullHandGate.Quality(Method, Score) : double.NaN;
            TargetSupported = declaredTarget == null ? false : targetSupported;
            TargetMatches = declaredTarget != null && targetSupported && ranked &&
                            string.Equals(Top1, declaredTarget, StringComparison.Ordinal);

            if (!ranked) Decision = FullHandDecisions.TechnicalFailure;
            else if (declaredTarget != null && !targetSupported)
                Decision = FullHandDecisions.UnsupportedTarget;
            else Decision = gate.AcceptsQuality(Quality)
                ? FullHandDecisions.Accepted : FullHandDecisions.Abstained;
        }

        public string[] CopyClasses() { return (string[])classes.Clone(); }
        public double[] CopyRawScores() { return (double[])rawScores.Clone(); }
        public string[] CopyClassesWithoutEvidence()
        {
            return (string[])classesWithoutEvidence.Clone();
        }
    }

    /// <summary>One bank's ranked results and gated decisions, with its reference bookkeeping.</summary>
    public sealed class FullHandBankOutcome
    {
        readonly FullHandMethodDecision[] decisions;

        public string BankId { get; private set; }
        public int EffectiveReferenceCount { get; private set; }
        public int CommonReferenceCount { get; private set; }
        public int ParticipantReferenceCount { get; private set; }
        public int ExcludedCommonReferenceCount { get; private set; }
        public int ExcludedParticipantReferenceCount { get; private set; }

        internal FullHandBankOutcome(string bankId, FullHandMethodDecision[] decisions,
                                     int commonUsed, int participantUsed,
                                     int excludedCommon, int excludedParticipant)
        {
            BankId = bankId;
            this.decisions = decisions;
            CommonReferenceCount = commonUsed;
            ParticipantReferenceCount = participantUsed;
            EffectiveReferenceCount = commonUsed + participantUsed;
            ExcludedCommonReferenceCount = excludedCommon;
            ExcludedParticipantReferenceCount = excludedParticipant;
        }

        public FullHandMethodDecision[] CopyDecisions()
        {
            return (FullHandMethodDecision[])decisions.Clone();
        }

        /// <summary>Selection by known method name over these same results.</summary>
        public FullHandMethodDecision Decision(string method)
        {
            return decisions[FullHandGateConfig.MethodIndex(FullHandGate.RequireMethod(method))];
        }
    }

    public sealed class FullHandRecognitionOutcome
    {
        public string QuerySampleId { get; private set; }
        public string QueryFeatureDigest { get; private set; }
        public string ParticipantId { get; private set; }
        public string CommonModelId { get; private set; }
        public string GateConfigId { get; private set; }
        public string DeclaredTarget { get; private set; }
        public bool TargetSupported { get; private set; }
        /// <summary>True when there were no usable additions and the one pass was reused.</summary>
        public bool ReusedCommonEvaluation { get; private set; }
        /// <summary>1 or 2. Observable proof that identical input is not scored twice.</summary>
        public int ScoringPasses { get; private set; }
        /// <summary>Common bank only.</summary>
        public FullHandBankOutcome Before { get; private set; }
        /// <summary>Common plus current-participant additions. RESEARCH ONLY.</summary>
        public FullHandBankOutcome After { get; private set; }

        internal FullHandRecognitionOutcome(FullHandSample query, string participantId,
            string commonModelId, string gateConfigId, string declaredTarget,
            bool targetSupported, bool reused, int passes,
            FullHandBankOutcome before, FullHandBankOutcome after)
        {
            QuerySampleId = query.SampleId;
            QueryFeatureDigest = query.FeatureDigest;
            ParticipantId = participantId;
            CommonModelId = commonModelId;
            GateConfigId = gateConfigId;
            DeclaredTarget = declaredTarget;
            TargetSupported = targetSupported;
            ReusedCommonEvaluation = reused;
            ScoringPasses = passes;
            Before = before;
            After = after;
        }
    }

    // ---- the composite scoring bank -----------------------------------------

    /// <summary>
    /// The common bank restricted to its allowed references, optionally followed by the
    /// current participant's allowed additions. It only re-indexes existing references; it
    /// owns no math.
    /// </summary>
    sealed class FullHandCompositeBank : IFullHandScoringBank
    {
        readonly IFullHandCommonBank common;
        readonly FullHandFeatureSequence[] additions;
        readonly int[][] byClass;
        readonly string[] vocabulary;

        internal FullHandCompositeBank(IFullHandCommonBank common, string[] vocabulary,
                                       bool[] commonAllowed,
                                       List<FullHandSample> allowedAdditions)
        {
            this.common = common;
            this.vocabulary = vocabulary;
            int commonCount = common.ReferenceCount;
            additions = new FullHandFeatureSequence[allowedAdditions.Count];
            var lists = new List<int>[vocabulary.Length];
            for (int index = 0; index < lists.Length; index++) lists[index] = new List<int>();
            for (int label = 0; label < vocabulary.Length; label++)
            {
                int[] indices = common.ReferencesForClass(label);
                for (int index = 0; index < indices.Length; index++)
                    if (commonAllowed[indices[index]]) lists[label].Add(indices[index]);
            }
            for (int index = 0; index < allowedAdditions.Count; index++)
            {
                FullHandSample sample = allowedAdditions[index];
                additions[index] = sample.Sequence;
                lists[LabelIndex(vocabulary, sample.Label)].Add(commonCount + index);
            }
            byClass = new int[vocabulary.Length][];
            for (int index = 0; index < lists.Length; index++) byClass[index] = lists[index].ToArray();
            UsedCommonCount = 0;
            for (int index = 0; index < commonCount; index++) if (commonAllowed[index]) UsedCommonCount++;
        }

        internal int UsedCommonCount { get; private set; }
        internal int UsedAdditionCount { get { return additions.Length; } }

        internal static int LabelIndex(string[] vocabulary, string label)
        {
            for (int index = 0; index < vocabulary.Length; index++)
                if (string.Equals(vocabulary[index], label, StringComparison.Ordinal)) return index;
            return -1;
        }

        public FullHandModel MlModel { get { return common.MlModel; } }
        public FullHandModel HybridModel { get { return common.HybridModel; } }
        public int ReferenceCount { get { return UsedCommonCount + additions.Length; } }
        public string[] CopyVocabulary() { return (string[])vocabulary.Clone(); }
        public int[] ReferencesForClass(int labelIndex) { return (int[])byClass[labelIndex].Clone(); }

        public FullHandFeatureSequence ReferenceSequence(int index)
        {
            int commonCount = common.ReferenceCount;
            return index < commonCount ? common.ReferenceSequence(index)
                                       : additions[index - commonCount];
        }
    }

    // ---- the pipeline --------------------------------------------------------

    public sealed class FullHandRecognitionPipeline
    {
        readonly IFullHandCommonBank common;
        readonly FullHandGateConfig gates;
        readonly FullHandParticipantReferenceSet currentReferences;
        readonly string[] vocabulary;
        readonly string[] commonReferenceIds;
        readonly string[] commonFeatureDigests;

        public string ParticipantId { get; private set; }
        public string CommonModelId { get { return common.ModelId; } }
        public string GateConfigId { get { return gates.ConfigId; } }
        public int CurrentReferenceCount { get { return currentReferences.Count; } }

        public FullHandRecognitionPipeline(FullHandRuntimeBundle commonBundle,
                                           FullHandGateConfig gates, string participantId,
                                           FullHandParticipantReferenceSet currentReferences)
            : this(new FullHandBundleCommonBank(commonBundle), gates, participantId,
                   currentReferences)
        {
        }

        public FullHandRecognitionPipeline(IFullHandCommonBank commonBank,
                                           FullHandGateConfig gates, string participantId,
                                           FullHandParticipantReferenceSet currentReferences)
        {
            if (commonBank == null) throw new ArgumentNullException("commonBank");
            if (gates == null) throw new ArgumentNullException("gates");
            this.common = commonBank;
            this.gates = gates;
            ParticipantId = FullHandPipelineText.RequireIdentifier(participantId, "participantId");
            this.currentReferences = currentReferences
                ?? FullHandParticipantReferenceSet.Empty(ParticipantId);
            if (!string.Equals(this.currentReferences.ParticipantId, ParticipantId,
                               StringComparison.Ordinal))
                throw new FullHandFormatException(FullHandPipelineCodes.ParticipantMismatch,
                    this.currentReferences.ParticipantId);

            vocabulary = commonBank.CopyVocabulary();
            for (int index = 0; index < this.currentReferences.Count; index++)
            {
                FullHandSample sample = this.currentReferences.At(index);
                if (FullHandCompositeBank.LabelIndex(vocabulary, sample.Label) < 0)
                    throw new FullHandFormatException(FullHandPipelineCodes.LabelUnsupported,
                        sample.Label);
            }
            commonReferenceIds = commonBank.CopyReferenceIds();
            if (commonReferenceIds.Length != commonBank.ReferenceCount)
                throw new FullHandFormatException(FullHandPipelineCodes.BankInvalid,
                    "the reference id table does not match the reference count");
            commonFeatureDigests = new string[commonReferenceIds.Length];
            for (int index = 0; index < commonReferenceIds.Length; index++)
                commonFeatureDigests[index] =
                    FullHandFeatureDigest.Compute(commonBank.ReferenceSequence(index));
        }

        public string[] CopyVocabulary() { return (string[])vocabulary.Clone(); }

        public bool SupportsLabel(string label)
        {
            return label != null && FullHandCompositeBank.LabelIndex(vocabulary, label) >= 0;
        }

        /// <summary>
        /// The single normalization point for a live capture: the 46-point capture becomes
        /// one feature sequence exactly once, and every method and both banks reuse it.
        /// </summary>
        public FullHandSample NormalizeCapture(FullHandCommonCapture capture, string sampleId,
                                               string sourceSha256)
        {
            if (capture == null) throw new ArgumentNullException("capture");
            return new FullHandSample(sampleId, ParticipantId, sourceSha256, null,
                                      FullHandPairGeometry.BuildFeatures(capture));
        }

        /// <summary>Primary entry: normalize once, then evaluate.</summary>
        public FullHandRecognitionOutcome Evaluate(FullHandCommonCapture capture, string sampleId,
                                                   string sourceSha256, string declaredTarget)
        {
            return EvaluateSample(NormalizeCapture(capture, sampleId, sourceSha256), declaredTarget);
        }

        /// <summary>
        /// Overload for evaluation work that already holds a normalized sample. The target,
        /// when declared, is consulted only AFTER scoring; it never reaches the geometry or
        /// the class scoring, and it can never force a class to be correct.
        /// </summary>
        public FullHandRecognitionOutcome EvaluateSample(FullHandSample query, string declaredTarget)
        {
            if (query == null) throw new ArgumentNullException("query");
            // a query belonging to somebody else is refused before any scoring, whether or
            // not this pipeline holds current-participant references
            if (!string.Equals(query.ParticipantId, ParticipantId, StringComparison.Ordinal))
                throw new FullHandFormatException(FullHandPipelineCodes.ParticipantMismatch,
                    query.SampleId);
            if (declaredTarget != null)
                FullHandPipelineText.RequireIdentifier(declaredTarget, "declaredTarget");
            bool targetSupported = SupportsLabel(declaredTarget);

            var commonAllowed = new bool[commonReferenceIds.Length];
            int excludedCommon = 0;
            for (int index = 0; index < commonAllowed.Length; index++)
            {
                // a common bundle has no source hashes; none are invented
                bool self = string.Equals(commonReferenceIds[index], query.SampleId,
                                          StringComparison.Ordinal)
                         || string.Equals(commonFeatureDigests[index], query.FeatureDigest,
                                          StringComparison.Ordinal);
                commonAllowed[index] = !self;
                if (self) excludedCommon++;
            }

            var allowedAdditions = new List<FullHandSample>();
            int excludedParticipant = 0;
            for (int index = 0; index < currentReferences.Count; index++)
            {
                FullHandSample sample = currentReferences.At(index);
                bool self = string.Equals(sample.SampleId, query.SampleId, StringComparison.Ordinal)
                         || string.Equals(sample.FeatureDigest, query.FeatureDigest,
                                          StringComparison.Ordinal)
                         || (sample.SourceSha256 != null && query.SourceSha256 != null &&
                             string.Equals(sample.SourceSha256, query.SourceSha256,
                                           StringComparison.Ordinal));
                if (self) excludedParticipant++;
                else allowedAdditions.Add(sample);
            }

            var empty = new List<FullHandSample>();
            var commonBank = new FullHandCompositeBank(common, vocabulary, commonAllowed, empty);
            FullHandEvaluation commonEvaluation = FullHandScorer.Evaluate(commonBank, query.Sequence);
            FullHandBankOutcome before = Snapshot(FullHandPipelineBanks.CommonOnly,
                commonEvaluation, declaredTarget, targetSupported,
                commonBank.UsedCommonCount, 0, excludedCommon, excludedParticipant);

            if (allowedAdditions.Count == 0)
            {
                // identical input, identical bank: the one pass is reused, never repeated
                FullHandBankOutcome after = Snapshot(FullHandPipelineBanks.CommonPlusParticipant,
                    commonEvaluation, declaredTarget, targetSupported,
                    commonBank.UsedCommonCount, 0, excludedCommon, excludedParticipant);
                return new FullHandRecognitionOutcome(query, ParticipantId, common.ModelId,
                    gates.ConfigId, declaredTarget, targetSupported, true, 1, before, after);
            }

            var personalBank = new FullHandCompositeBank(common, vocabulary, commonAllowed,
                                                         allowedAdditions);
            FullHandEvaluation personalEvaluation =
                FullHandScorer.Evaluate(personalBank, query.Sequence);
            FullHandBankOutcome researchAfter = Snapshot(FullHandPipelineBanks.CommonPlusParticipant,
                personalEvaluation, declaredTarget, targetSupported,
                personalBank.UsedCommonCount, personalBank.UsedAdditionCount,
                excludedCommon, excludedParticipant);
            return new FullHandRecognitionOutcome(query, ParticipantId, common.ModelId,
                gates.ConfigId, declaredTarget, targetSupported, false, 2, before, researchAfter);
        }

        FullHandBankOutcome Snapshot(string bankId, FullHandEvaluation evaluation,
                                     string declaredTarget, bool targetSupported,
                                     int commonUsed, int participantUsed,
                                     int excludedCommon, int excludedParticipant)
        {
            var decisions = new FullHandMethodDecision[FullHandMethods.All.Length];
            for (int index = 0; index < decisions.Length; index++)
            {
                string method = FullHandMethods.All[index];
                decisions[index] = new FullHandMethodDecision(evaluation.Method(method),
                    gates.Gate(method), declaredTarget, targetSupported);
            }
            return new FullHandBankOutcome(bankId, decisions, commonUsed, participantUsed,
                                           excludedCommon, excludedParticipant);
        }
    }
}
