// M3_H3_RUNTIME_R1 - versioned local bundle reader for the frozen H3 v1 artifact.
// Prepared in the workspace. NOT installed and NOT added to StreamingAssets.
//
// Dependencies: mscorlib and System only. System.Security.Cryptography.SHA256 lives in
// mscorlib, so csc needs no extra /reference and Unity needs no package. The manifest is
// deliberately a strict line format rather than JSON precisely so that no JSON library,
// no package and no UnityEngine reference is required to parse it.
//
// Bundle layout, written by export_h3_runtime_r1.py:
//   h3_runtime_manifest.txt   key = value lines, plus one "array = ..." line per array
//   h3_runtime_payload.bin    little-endian float64 and uint8, concatenated, no padding
//
// A raw little-endian payload is used so every float64 bit pattern, including any NaN a
// masked slot might carry, survives the round trip exactly. Nothing is reformatted.
//
// The bundle carries the frozen thresholds, and the loader REQUIRES them to equal the
// compiled-in constants. A bundle therefore cannot quietly lower a gate.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace WordsInSilence.Recognition.H3Runtime
{
    /// <summary>A typed bundle failure. Never substituted with a usable value.</summary>
    public sealed class H3FormatException : Exception
    {
        public string Code { get; private set; }

        public H3FormatException(string code, string message)
            : base(code + ": " + message)
        {
            Code = code;
        }
    }

    public static class H3FormatCodes
    {
        public const string ManifestMissing = "MANIFEST_MISSING";
        public const string PayloadMissing = "PAYLOAD_MISSING";
        public const string MalformedLine = "MALFORMED_MANIFEST_LINE";
        public const string MissingKey = "MISSING_MANIFEST_KEY";
        public const string UnsupportedFormat = "UNSUPPORTED_FORMAT_VERSION";
        public const string UnsupportedEncoding = "UNSUPPORTED_ENCODING";
        public const string VersionMismatch = "VERSION_MISMATCH";
        public const string FeatureOrderMismatch = "FEATURE_ORDER_MISMATCH";
        public const string ThresholdMismatch = "THRESHOLD_MISMATCH";
        public const string VocabularyInvalid = "VOCABULARY_INVALID";
        public const string BankClassMissing = "BANK_CLASS_MISSING";
        public const string ArrayMissing = "ARRAY_MISSING";
        public const string ArrayShapeInvalid = "ARRAY_SHAPE_INVALID";
        public const string ArrayRangeInvalid = "ARRAY_RANGE_INVALID";
        public const string ArrayOverlap = "ARRAY_OVERLAP";
        public const string PayloadSizeMismatch = "PAYLOAD_SIZE_MISMATCH";
        public const string DigestMismatch = "DIGEST_MISMATCH";
        public const string NonfiniteCoefficient = "NONFINITE_COEFFICIENT";
        public const string NonpositiveScale = "NONPOSITIVE_SCALE";
        public const string NonfiniteActiveCoordinate = "NONFINITE_ACTIVE_COORDINATE";
        public const string MaskValueInvalid = "MASK_VALUE_INVALID";
    }

    public sealed class H3ReferenceSequence
    {
        public double[] Coordinates { get; private set; }
        public bool[] Mask { get; private set; }
        public int FrameCount { get; private set; }

        public H3ReferenceSequence(double[] coordinates, bool[] mask, int frameCount)
        {
            Coordinates = coordinates; Mask = mask; FrameCount = frameCount;
        }
    }

    /// <summary>The loaded, validated bundle. Immutable once constructed.</summary>
    public sealed class H3RuntimeBundle
    {
        public const string SupportedFormat = "wis_h3_runtime_bundle_v1";

        public string SourceArtifactSchema { get; private set; }
        public string SourceArtifactName { get; private set; }
        public string SourceNpzSha256 { get; private set; }
        public string SourceJsonSha256 { get; private set; }
        public string PayloadSha256 { get; private set; }
        public IDictionary<string, string> Versions { get; private set; }
        public string[] Vocabulary { get; private set; }
        public string[] PairClassOrder { get; private set; }
        public string[] FeatureOrder { get; private set; }

        public double[] ScalerMean { get; private set; }
        public double[] ScalerScale { get; private set; }
        public double[] Coefficients { get; private set; }
        public double Intercept { get; private set; }
        public IDictionary<string, H3ReferenceSequence[]> Bank { get; private set; }

        internal H3RuntimeBundle(string schema, string name, string npz, string json,
            string payloadDigest, IDictionary<string, string> versions, string[] vocabulary,
            string[] pairClassOrder, string[] featureOrder, double[] scalerMean,
            double[] scalerScale, double[] coefficients, double intercept,
            IDictionary<string, H3ReferenceSequence[]> bank)
        {
            SourceArtifactSchema = schema; SourceArtifactName = name;
            SourceNpzSha256 = npz; SourceJsonSha256 = json; PayloadSha256 = payloadDigest;
            Versions = versions; Vocabulary = vocabulary; PairClassOrder = pairClassOrder;
            FeatureOrder = featureOrder; ScalerMean = scalerMean; ScalerScale = scalerScale;
            Coefficients = coefficients; Intercept = intercept; Bank = bank;
        }
    }

    public static class H3RuntimeFormat
    {
        public const string ManifestFileName = "h3_runtime_manifest.txt";
        public const string PayloadFileName = "h3_runtime_payload.bin";

        sealed class ArrayEntry
        {
            public string Name;
            public string Dtype;
            public int[] Shape;
            public long Offset;
            public long Length;
            public string Digest;
        }

        static void Fail(string code, string message)
        {
            throw new H3FormatException(code, message);
        }

        static string Require(IDictionary<string, string> values, string key)
        {
            string value;
            if (!values.TryGetValue(key, out value) || value.Length == 0)
                Fail(H3FormatCodes.MissingKey, key);
            return values[key];
        }

        static double RequireDouble(IDictionary<string, string> values, string key)
        {
            double parsed;
            if (!double.TryParse(Require(values, key), NumberStyles.Float,
                                 CultureInfo.InvariantCulture, out parsed))
                Fail(H3FormatCodes.MalformedLine, key + " is not a number");
            return parsed;
        }

        static string[] SplitList(string value)
        {
            if (value.Length == 0) return new string[0];
            string[] parts = value.Split(',');
            for (int index = 0; index < parts.Length; index++) parts[index] = parts[index].Trim();
            return parts;
        }

        static string Hex(byte[] data, int offset, int count)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(data, offset, count);
                var text = new StringBuilder(digest.Length * 2);
                for (int index = 0; index < digest.Length; index++)
                    text.Append(digest[index].ToString("x2", CultureInfo.InvariantCulture));
                return text.ToString();
            }
        }

        static double ReadDouble(byte[] payload, long at)
        {
            if (BitConverter.IsLittleEndian) return BitConverter.ToDouble(payload, (int)at);
            var swapped = new byte[8];
            for (int index = 0; index < 8; index++) swapped[index] = payload[at + 7 - index];
            return BitConverter.ToDouble(swapped, 0);
        }

        /// <summary>
        /// Loads and fully validates a bundle. Every size, range and digest is checked
        /// before any numeric array is materialized, and every numeric contract is checked
        /// before the bundle is returned.
        /// </summary>
        public static H3RuntimeBundle Load(string directory)
        {
            string manifestPath = Path.Combine(directory, ManifestFileName);
            string payloadPath = Path.Combine(directory, PayloadFileName);
            if (!File.Exists(manifestPath)) Fail(H3FormatCodes.ManifestMissing, manifestPath);
            if (!File.Exists(payloadPath)) Fail(H3FormatCodes.PayloadMissing, payloadPath);

            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            var arrays = new List<ArrayEntry>();
            string[] lines = File.ReadAllLines(manifestPath, new UTF8Encoding(false));
            for (int number = 0; number < lines.Length; number++)
            {
                string line = lines[number].Trim();
                if (line.Length == 0 || line[0] == '#') continue;
                int separator = line.IndexOf('=');
                if (separator <= 0)
                    Fail(H3FormatCodes.MalformedLine, "line " + (number + 1));
                string key = line.Substring(0, separator).Trim();
                string value = line.Substring(separator + 1).Trim();
                if (key == "array") arrays.Add(ParseArray(value, number + 1));
                else if (values.ContainsKey(key))
                    Fail(H3FormatCodes.MalformedLine, "duplicate key " + key);
                else values[key] = value;
            }

            if (Require(values, "format") != H3RuntimeBundle.SupportedFormat)
                Fail(H3FormatCodes.UnsupportedFormat, values["format"]);
            if (Require(values, "endianness") != "little")
                Fail(H3FormatCodes.UnsupportedEncoding, "endianness " + values["endianness"]);
            if (Require(values, "float_dtype") != "float64")
                Fail(H3FormatCodes.UnsupportedEncoding, "float dtype " + values["float_dtype"]);
            if (Require(values, "mask_dtype") != "uint8")
                Fail(H3FormatCodes.UnsupportedEncoding, "mask dtype " + values["mask_dtype"]);

            // pinned geometry and gates: a bundle may not move them
            if ((int)RequireDouble(values, "landmark_count") != H3Constants.LandmarkCount)
                Fail(H3FormatCodes.VersionMismatch, "landmark_count");
            if ((int)RequireDouble(values, "resample_frames") != H3Constants.ResampleFrames)
                Fail(H3FormatCodes.VersionMismatch, "resample_frames");
            if ((int)RequireDouble(values, "feature_count") != H3Constants.PairFeatureCount)
                Fail(H3FormatCodes.FeatureOrderMismatch, "feature_count");
            if (RequireDouble(values, "relative_accept_probability")
                != H3Decision.RelativeAcceptProbability)
                Fail(H3FormatCodes.ThresholdMismatch, "relative_accept_probability");
            if (RequireDouble(values, "relative_accept_margin")
                != H3Decision.RelativeAcceptMargin)
                Fail(H3FormatCodes.ThresholdMismatch, "relative_accept_margin");
            if (RequireDouble(values, "absolute_compatibility_floor")
                != H3Decision.AbsoluteCompatibilityFloor)
                Fail(H3FormatCodes.ThresholdMismatch, "absolute_compatibility_floor");

            string[] featureOrder = SplitList(Require(values, "feature_order"));
            string[] expectedOrder = H3PairGeometry.FeatureNames();
            if (featureOrder.Length != expectedOrder.Length)
                Fail(H3FormatCodes.FeatureOrderMismatch, "length");
            for (int index = 0; index < expectedOrder.Length; index++)
                if (featureOrder[index] != expectedOrder[index])
                    Fail(H3FormatCodes.FeatureOrderMismatch, "position " + index);

            string[] pairClassOrder = SplitList(Require(values, "pair_class_order"));
            if (pairClassOrder.Length != 2 || pairClassOrder[0] != "different_word"
                || pairClassOrder[1] != "same_word")
                Fail(H3FormatCodes.VersionMismatch, "pair_class_order");

            string[] vocabulary = SplitList(Require(values, "vocabulary"));
            if (vocabulary.Length == 0)
                Fail(H3FormatCodes.VocabularyInvalid, "the vocabulary is empty");
            var seen = new Dictionary<string, bool>(StringComparer.Ordinal);
            for (int index = 0; index < vocabulary.Length; index++)
            {
                if (vocabulary[index].Length == 0)
                    Fail(H3FormatCodes.VocabularyInvalid, "an empty class name");
                if (seen.ContainsKey(vocabulary[index]))
                    Fail(H3FormatCodes.VocabularyInvalid, "a repeated class name");
                seen[vocabulary[index]] = true;
            }

            var versions = new Dictionary<string, string>(StringComparer.Ordinal);
            string[] versionKeys = { "coord_contract", "alignment", "pair_features",
                                     "normalization", "model" };
            for (int index = 0; index < versionKeys.Length; index++)
                versions[versionKeys[index]] = Require(values, "version_" + versionKeys[index]);

            // ---- array table checks, all before any numeric array is built
            long declaredPayload = (long)RequireDouble(values, "payload_bytes");
            var payloadInfo = new FileInfo(payloadPath);
            if (payloadInfo.Length != declaredPayload)
                Fail(H3FormatCodes.PayloadSizeMismatch,
                     payloadInfo.Length + " on disk, " + declaredPayload + " declared");

            var table = new Dictionary<string, ArrayEntry>(StringComparer.Ordinal);
            long covered = 0;
            long previousEnd = 0;
            for (int index = 0; index < arrays.Count; index++)
            {
                ArrayEntry entry = arrays[index];
                if (table.ContainsKey(entry.Name))
                    Fail(H3FormatCodes.MalformedLine, "duplicate array " + entry.Name);
                int elementSize = entry.Dtype == "float64" ? 8
                                  : entry.Dtype == "uint8" ? 1 : 0;
                if (elementSize == 0)
                    Fail(H3FormatCodes.UnsupportedEncoding, "dtype " + entry.Dtype);
                long elements = 1;
                for (int axis = 0; axis < entry.Shape.Length; axis++)
                {
                    if (entry.Shape[axis] <= 0)
                        Fail(H3FormatCodes.ArrayShapeInvalid, entry.Name);
                    elements *= entry.Shape[axis];
                }
                if (elements * elementSize != entry.Length)
                    Fail(H3FormatCodes.ArrayShapeInvalid,
                         entry.Name + " length disagrees with its shape");
                if (entry.Offset < 0 || entry.Length < 0
                    || entry.Offset + entry.Length > declaredPayload)
                    Fail(H3FormatCodes.ArrayRangeInvalid, entry.Name);
                if (entry.Offset < previousEnd)
                    Fail(H3FormatCodes.ArrayOverlap, entry.Name);
                previousEnd = entry.Offset + entry.Length;
                covered += entry.Length;
                table[entry.Name] = entry;
            }
            if (covered != declaredPayload)
                Fail(H3FormatCodes.PayloadSizeMismatch,
                     "the arrays cover " + covered + " of " + declaredPayload + " bytes");

            byte[] payload = File.ReadAllBytes(payloadPath);
            if (Hex(payload, 0, payload.Length) != Require(values, "payload_sha256"))
                Fail(H3FormatCodes.DigestMismatch, "payload");
            foreach (KeyValuePair<string, ArrayEntry> pair in table)
            {
                ArrayEntry entry = pair.Value;
                if (Hex(payload, (int)entry.Offset, (int)entry.Length) != entry.Digest)
                    Fail(H3FormatCodes.DigestMismatch, entry.Name);
            }

            // ---- materialize and check the numeric contracts
            double[] scalerMean = Doubles(table, payload, "scaler_mean",
                                          H3Constants.PairFeatureCount);
            double[] scalerScale = Doubles(table, payload, "scaler_scale",
                                           H3Constants.PairFeatureCount);
            double[] coefficients = Doubles(table, payload, "coef",
                                            H3Constants.PairFeatureCount);
            double[] intercept = Doubles(table, payload, "intercept", 1);
            CheckFinite(scalerMean, "scaler_mean");
            CheckFinite(scalerScale, "scaler_scale");
            CheckFinite(coefficients, "coef");
            CheckFinite(intercept, "intercept");
            for (int index = 0; index < scalerScale.Length; index++)
                if (!(scalerScale[index] > 0.0))
                    Fail(H3FormatCodes.NonpositiveScale, "index " + index);

            var bank = new Dictionary<string, H3ReferenceSequence[]>(StringComparer.Ordinal);
            string[] counts = SplitList(Require(values, "bank_reference_counts"));
            var declaredCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int index = 0; index < counts.Length; index++)
            {
                string[] parts = counts[index].Split(':');
                // definite assignment: the || short-circuits when parts.Length != 2, so
                // TryParse may never run and 'parsed' would be read unassigned by the
                // range test and by the assignment below. Fail() always throws, but the
                // compiler does not model that, so the local is initialized. No refusal,
                // no message and no accepted value changes.
                int parsed = 0;
                if (parts.Length != 2 || !int.TryParse(parts[1], NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out parsed) || parsed <= 0)
                    Fail(H3FormatCodes.MalformedLine, "bank_reference_counts");
                declaredCounts[parts[0].Trim()] = parsed;
            }
            for (int index = 0; index < vocabulary.Length; index++)
            {
                string label = vocabulary[index];
                if (!declaredCounts.ContainsKey(label))
                    Fail(H3FormatCodes.BankClassMissing, label);
                int references = declaredCounts[label];
                var sequences = new H3ReferenceSequence[references];
                for (int reference = 0; reference < references; reference++)
                {
                    string coordinateName = "bank_" + label + "_" + reference + "_coords";
                    string maskName = "bank_" + label + "_" + reference + "_mask";
                    int frames = H3Constants.ResampleFrames;
                    int landmarks = H3Constants.LandmarkCount;
                    double[] coordinates = Doubles(table, payload, coordinateName,
                        frames * landmarks * H3Constants.CoordDims);
                    bool[] mask = Mask(table, payload, maskName, frames * landmarks);
                    for (int slot = 0; slot < mask.Length; slot++)
                    {
                        if (!mask[slot]) continue;
                        int at = slot * H3Constants.CoordDims;
                        for (int channel = 0; channel < H3Constants.CoordDims; channel++)
                        {
                            double value = coordinates[at + channel];
                            if (double.IsNaN(value) || double.IsInfinity(value))
                                Fail(H3FormatCodes.NonfiniteActiveCoordinate, coordinateName);
                        }
                    }
                    sequences[reference] = new H3ReferenceSequence(coordinates, mask, frames);
                }
                bank[label] = sequences;
            }

            string json;
            values.TryGetValue("source_json_sha256", out json);
            return new H3RuntimeBundle(
                Require(values, "source_artifact_schema"),
                Require(values, "source_artifact_name"),
                Require(values, "source_npz_sha256"), json ?? string.Empty,
                values["payload_sha256"], versions, vocabulary, pairClassOrder,
                featureOrder, scalerMean, scalerScale, coefficients, intercept[0], bank);
        }

        static ArrayEntry ParseArray(string value, int lineNumber)
        {
            string[] parts = value.Split('|');
            if (parts.Length != 6)
                Fail(H3FormatCodes.MalformedLine, "array line " + lineNumber);
            var entry = new ArrayEntry();
            entry.Name = parts[0].Trim();
            entry.Dtype = parts[1].Trim();
            string[] shape = parts[2].Trim().Split(',');
            entry.Shape = new int[shape.Length];
            for (int axis = 0; axis < shape.Length; axis++)
                if (!int.TryParse(shape[axis], NumberStyles.Integer,
                                  CultureInfo.InvariantCulture, out entry.Shape[axis]))
                    Fail(H3FormatCodes.ArrayShapeInvalid, entry.Name);
            if (!long.TryParse(parts[3].Trim(), NumberStyles.Integer,
                               CultureInfo.InvariantCulture, out entry.Offset)
                || !long.TryParse(parts[4].Trim(), NumberStyles.Integer,
                                  CultureInfo.InvariantCulture, out entry.Length))
                Fail(H3FormatCodes.ArrayRangeInvalid, entry.Name);
            entry.Digest = parts[5].Trim();
            if (entry.Name.Length == 0 || entry.Digest.Length != 64)
                Fail(H3FormatCodes.MalformedLine, "array line " + lineNumber);
            return entry;
        }

        static double[] Doubles(IDictionary<string, ArrayEntry> table, byte[] payload,
                                string name, int expectedElements)
        {
            ArrayEntry entry;
            if (!table.TryGetValue(name, out entry)) Fail(H3FormatCodes.ArrayMissing, name);
            entry = table[name];
            if (entry.Dtype != "float64")
                Fail(H3FormatCodes.UnsupportedEncoding, name + " is not float64");
            if (entry.Length != (long)expectedElements * 8)
                Fail(H3FormatCodes.ArrayShapeInvalid, name);
            var result = new double[expectedElements];
            for (int index = 0; index < expectedElements; index++)
                result[index] = ReadDouble(payload, entry.Offset + index * 8L);
            return result;
        }

        static bool[] Mask(IDictionary<string, ArrayEntry> table, byte[] payload,
                           string name, int expectedElements)
        {
            ArrayEntry entry;
            if (!table.TryGetValue(name, out entry)) Fail(H3FormatCodes.ArrayMissing, name);
            entry = table[name];
            if (entry.Dtype != "uint8")
                Fail(H3FormatCodes.UnsupportedEncoding, name + " is not uint8");
            if (entry.Length != expectedElements)
                Fail(H3FormatCodes.ArrayShapeInvalid, name);
            var result = new bool[expectedElements];
            for (int index = 0; index < expectedElements; index++)
            {
                byte raw = payload[entry.Offset + index];
                if (raw > 1) Fail(H3FormatCodes.MaskValueInvalid, name);
                result[index] = raw == 1;
            }
            return result;
        }

        static void CheckFinite(double[] values, string name)
        {
            for (int index = 0; index < values.Length; index++)
                if (double.IsNaN(values[index]) || double.IsInfinity(values[index]))
                    Fail(H3FormatCodes.NonfiniteCoefficient, name);
        }
    }
}
