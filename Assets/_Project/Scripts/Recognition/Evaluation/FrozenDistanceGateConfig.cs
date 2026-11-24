using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace WordsInSilence.Recognition.Evaluation
{
    /// <summary>Validated view of the immutable Step 8C canonical threshold payload.</summary>
    public sealed class FrozenDistanceGateConfig
    {
        public string Version { get; }
        public string RunId { get; }
        public string SelectionScope { get; }
        public string ComparisonOperator { get; }
        public string Formula { get; }
        public double ValidationMaxTrue { get; }
        public double ValidationMinWrong { get; }
        public double ValidationSeparationGap { get; }
        public double Threshold { get; }
        public IReadOnlyList<string> ThresholdSelectionQueryIds { get; }
        public string CanonicalPayloadSha256 { get; }
        public string Step8ADeliverySha256 { get; }
        public string Step8BDeliverySha256 { get; }
        public string Step8ARunId { get; }
        public string Step8BRunId { get; }
        public bool NegativeOrUnknownSamplesAvailable { get; }
        public string OpenSetRejectionPerformance { get; }

        internal FrozenDistanceGateConfig(
            string version,
            string runId,
            string selectionScope,
            string comparisonOperator,
            string formula,
            double validationMaxTrue,
            double validationMinWrong,
            double validationSeparationGap,
            double threshold,
            IReadOnlyList<string> thresholdSelectionQueryIds,
            string canonicalPayloadSha256,
            string step8ADeliverySha256,
            string step8BDeliverySha256,
            string step8ARunId,
            string step8BRunId,
            bool negativeOrUnknownSamplesAvailable,
            string openSetRejectionPerformance)
        {
            Version = version;
            RunId = runId;
            SelectionScope = selectionScope;
            ComparisonOperator = comparisonOperator;
            Formula = formula;
            ValidationMaxTrue = validationMaxTrue;
            ValidationMinWrong = validationMinWrong;
            ValidationSeparationGap = validationSeparationGap;
            Threshold = threshold;
            ThresholdSelectionQueryIds = thresholdSelectionQueryIds;
            CanonicalPayloadSha256 = canonicalPayloadSha256;
            Step8ADeliverySha256 = step8ADeliverySha256;
            Step8BDeliverySha256 = step8BDeliverySha256;
            Step8ARunId = step8ARunId;
            Step8BRunId = step8BRunId;
            NegativeOrUnknownSamplesAvailable = negativeOrUnknownSamplesAvailable;
            OpenSetRejectionPerformance = openSetRejectionPerformance;
        }
    }

    /// <summary>Strict loader that verifies the exact canonical payload bytes before exposing the gate.</summary>
    public static class FrozenDistanceGateConfigLoader
    {
        const string Prefix = "{\"canonical_payload\":";
        const string HashMarker = ",\"canonical_payload_sha256\":\"";
        static readonly Regex LowerSha256 = new Regex("^[0-9a-f]{64}$", RegexOptions.CultureInvariant);

        public static FrozenDistanceGateConfig LoadFile(string path, IEnumerable<string> sealedTestIds)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path is required", nameof(path));
            return Load(File.ReadAllBytes(path), sealedTestIds);
        }

        public static FrozenDistanceGateConfig Load(byte[] fileBytes, IEnumerable<string> sealedTestIds)
        {
            if (fileBytes == null) throw new ArgumentNullException(nameof(fileBytes));
            if (fileBytes.Length < 100 || fileBytes[fileBytes.Length - 1] != (byte)'\n' ||
                (fileBytes.Length > 1 && fileBytes[fileBytes.Length - 2] == (byte)'\n'))
                throw new InvalidDataException("threshold config must have exactly one terminal LF");

            string wrapper;
            try
            {
                wrapper = new UTF8Encoding(false, true).GetString(fileBytes, 0, fileBytes.Length - 1);
            }
            catch (DecoderFallbackException exc)
            {
                throw new InvalidDataException("threshold config is not valid UTF-8", exc);
            }
            if (!wrapper.StartsWith(Prefix, StringComparison.Ordinal) || !wrapper.EndsWith("\"}", StringComparison.Ordinal))
                throw new InvalidDataException("threshold config wrapper is malformed");
            int markerIndex = wrapper.LastIndexOf(HashMarker, StringComparison.Ordinal);
            if (markerIndex <= Prefix.Length) throw new InvalidDataException("threshold config hash field is missing");
            string canonicalPayload = wrapper.Substring(Prefix.Length, markerIndex - Prefix.Length);
            string suppliedHash = wrapper.Substring(markerIndex + HashMarker.Length, wrapper.Length - markerIndex - HashMarker.Length - 2);
            if (!LowerSha256.IsMatch(suppliedHash)) throw new InvalidDataException("threshold config hash is malformed");
            string actualHash = Sha256(Encoding.UTF8.GetBytes(canonicalPayload));
            if (!string.Equals(actualHash, suppliedHash, StringComparison.Ordinal))
                throw new InvalidDataException("threshold canonical payload SHA-256 mismatch");

            string version = RequiredString(canonicalPayload, "version");
            string runId = RequiredString(canonicalPayload, "run_id");
            string scope = RequiredString(canonicalPayload, "selection_scope");
            string comparison = RequiredString(canonicalPayload, "comparison_operator");
            string formula = RequiredString(canonicalPayload, "formula");
            string openSet = RequiredString(canonicalPayload, "open_set_rejection_performance");
            string step8AHash = RequiredString(canonicalPayload, "step8a_delivery_sha256");
            string step8BHash = RequiredString(canonicalPayload, "step8b_delivery_sha256");
            string step8ARunId = RequiredString(canonicalPayload, "step8a_run_id");
            string step8BRunId = RequiredString(canonicalPayload, "step8b_run_id");
            double maxTrue = RequiredDouble(canonicalPayload, "validation_max_true");
            double minWrong = RequiredDouble(canonicalPayload, "validation_min_wrong");
            double gap = RequiredDouble(canonicalPayload, "validation_separation_gap");
            double threshold = RequiredDouble(canonicalPayload, "threshold");
            bool negatives = RequiredBool(canonicalPayload, "negative_or_unknown_samples_available");
            string[] selectionIds = RequiredStringArray(canonicalPayload, "threshold_selection_query_ids");

            if (version != "1.0" || scope != "validation_only" || comparison != "<=" ||
                openSet != "not_evaluated" || negatives || selectionIds.Length != 9)
                throw new InvalidDataException("threshold policy fields are invalid");
            if (!LowerSha256.IsMatch(step8AHash) || !LowerSha256.IsMatch(step8BHash))
                throw new InvalidDataException("threshold provenance hash is invalid");
            if (!(gap > 0.0) || Math.Abs((minWrong - maxTrue) - gap) > 1e-15 ||
                Math.Abs(((maxTrue + minWrong) / 2.0) - threshold) > 1e-15)
                throw new InvalidDataException("threshold formula values are inconsistent");
            ValidateSelectionDoesNotUseTestIds(selectionIds, sealedTestIds);

            return new FrozenDistanceGateConfig(
                version, runId, scope, comparison, formula, maxTrue, minWrong, gap, threshold,
                Array.AsReadOnly(selectionIds), suppliedHash, step8AHash, step8BHash,
                step8ARunId, step8BRunId, negatives, openSet);
        }

        public static void ValidateSelectionDoesNotUseTestIds(IEnumerable<string> selectionIds, IEnumerable<string> sealedTestIds)
        {
            if (selectionIds == null) throw new ArgumentNullException(nameof(selectionIds));
            if (sealedTestIds == null) throw new ArgumentNullException(nameof(sealedTestIds));
            var sealedSet = new HashSet<string>(sealedTestIds, StringComparer.Ordinal);
            if (selectionIds.Any(sealedSet.Contains))
                throw new InvalidDataException("threshold selection provenance contains a sealed test ID");
        }

        static string RequiredString(string payload, string name)
        {
            Match match = Regex.Match(payload, "\\\"" + Regex.Escape(name) + "\\\":\\\"([^\\\"]+)\\\"", RegexOptions.CultureInvariant);
            if (!match.Success) throw new InvalidDataException($"threshold field {name} is missing");
            return match.Groups[1].Value;
        }

        static double RequiredDouble(string payload, string name)
        {
            Match match = Regex.Match(payload, "\\\"" + Regex.Escape(name) + "\\\":(-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", RegexOptions.CultureInvariant);
            if (!match.Success || !double.TryParse(match.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ||
                double.IsNaN(value) || double.IsInfinity(value))
                throw new InvalidDataException($"threshold numeric field {name} is invalid");
            return value;
        }

        static bool RequiredBool(string payload, string name)
        {
            Match match = Regex.Match(payload, "\\\"" + Regex.Escape(name) + "\\\":(true|false)", RegexOptions.CultureInvariant);
            if (!match.Success) throw new InvalidDataException($"threshold Boolean field {name} is missing");
            return match.Groups[1].Value == "true";
        }

        static string[] RequiredStringArray(string payload, string name)
        {
            Match match = Regex.Match(payload, "\\\"" + Regex.Escape(name) + "\\\":\\[([^\\]]*)\\]", RegexOptions.CultureInvariant);
            if (!match.Success) throw new InvalidDataException($"threshold array field {name} is missing");
            MatchCollection values = Regex.Matches(match.Groups[1].Value, "\\\"([^\\\"]+)\\\"", RegexOptions.CultureInvariant);
            var result = values.Cast<Match>().Select(item => item.Groups[1].Value).ToArray();
            string reconstructed = string.Join(",", result.Select(value => "\"" + value + "\""));
            if (!string.Equals(reconstructed, match.Groups[1].Value, StringComparison.Ordinal))
                throw new InvalidDataException($"threshold array field {name} is malformed");
            return result;
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create())
            {
                byte[] hash = sha.ComputeHash(bytes);
                var builder = new StringBuilder(hash.Length * 2);
                foreach (byte value in hash) builder.Append(value.ToString("x2"));
                return builder.ToString();
            }
        }
    }
}
