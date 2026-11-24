// F4 - private local runtime bundle: JSON manifest + little-endian binary payload.
//
// The loader is pure: it reads exactly the manifest path it is given. No auto-search, no
// default location, no Resources, no StreamingAssets, no asset or model download, and no
// startup integration. Reference IDs are opaque; the raw reference arrays stay private
// even so.
//
// Payload layout, all little-endian, strings length-prefixed UTF8 (int32 byte count):
//   "WISFHB01" | int32 formatVersion | int32 frames, points, dims, fixedWidth, alignedWidth
//   int32 vocabularyCount   { string label }
//   model x2 (ml then hybrid): int32 width | double[width] mean, scale, coefficients | double intercept
//   int32 referenceCount    { string id | int32 labelIndex | double[frames*points*3] | byte[frames*points] }
//   end of file, with no trailing byte
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    [Serializable]
    public sealed class FullHandBundleManifest
    {
        public string schema;
        public int format_version;
        public bool synthetic;
        public string payload_file;
        public long payload_bytes;
        public string payload_sha256;
        public string point_schema;
        public int resample_frames;
        public int feature_point_count;
        public int coord_dims;
        public int fixed_feature_count;
        public int aligned_feature_count;
        public int vocabulary_count;
        public int reference_count;
        public string[] vocabulary;
        public string[] reference_ids;
        public int[] reference_label_indices;
        public string exporter_version;
        public string provenance_note;
        public string[] provenance_hashes;
        public string handling;
    }

    public static class FullHandBundleCodes
    {
        public const string ManifestMissing = "MANIFEST_MISSING";
        public const string ManifestMalformed = "MANIFEST_MALFORMED";
        public const string PayloadMissing = "PAYLOAD_MISSING";
        public const string PayloadPathInvalid = "PAYLOAD_PATH_INVALID";
        public const string SchemaMismatch = "BUNDLE_SCHEMA_MISMATCH";
        public const string DimensionMismatch = "BUNDLE_DIMENSION_MISMATCH";
        public const string SizeMismatch = "PAYLOAD_SIZE_MISMATCH";
        public const string DigestMismatch = "PAYLOAD_DIGEST_MISMATCH";
        public const string HeaderInvalid = "PAYLOAD_HEADER_INVALID";
        public const string Truncated = "PAYLOAD_TRUNCATED";
        public const string TrailingBytes = "PAYLOAD_TRAILING_BYTES";
        public const string CountInvalid = "BUNDLE_COUNT_INVALID";
        public const string LabelInvalid = "BUNDLE_LABEL_INVALID";
        public const string ReferenceInvalid = "BUNDLE_REFERENCE_INVALID";
        public const string ModelInvalid = "BUNDLE_MODEL_INVALID";
        public const string ManifestPayloadDisagree = "MANIFEST_PAYLOAD_DISAGREE";
    }

    public sealed class FullHandReference
    {
        public string Id { get; private set; }
        public int LabelIndex { get; private set; }
        public FullHandFeatureSequence Sequence { get; private set; }

        internal FullHandReference(string id, int labelIndex, FullHandFeatureSequence sequence)
        {
            Id = id; LabelIndex = labelIndex; Sequence = sequence;
        }
    }

    public sealed class FullHandRuntimeBundle : IFullHandScoringBank
    {
        public const string SupportedSchema = "wis_full_hand_runtime_bundle_v1";
        public const int SupportedFormatVersion = 1;
        // Staged research candidate. Reading it requires an explicit phase manifest, so the
        // shipped launcher keeps loading the unchanged v1 bundle.
        public const string PhaseSchema = "wis_full_hand_phase_runtime_bundle_v1";
        public const int PhaseFormatVersion = 2;
        public const int MaxVocabulary = 1024;
        public const int MaxReferences = 2048;
        public const int MaxStringBytes = 256;
        public const long MaxPayloadBytes = 256L * 1024L * 1024L;
        static readonly byte[] Magic = Encoding.ASCII.GetBytes("WISFHB01");
        static readonly byte[] PhaseMagic = Encoding.ASCII.GetBytes("WISFHB02");

        readonly string[] vocabulary;
        readonly FullHandReference[] references;
        readonly int[][] referencesByClass;

        public bool Synthetic { get; private set; }
        public string PayloadSha256 { get; private set; }
        public string ExporterVersion { get; private set; }
        public FullHandModel MlModel { get; private set; }
        public FullHandModel HybridModel { get; private set; }
        public int ReferenceCount { get { return references.Length; } }
        public int VocabularyCount { get { return vocabulary.Length; } }

        FullHandRuntimeBundle(string[] vocabulary, FullHandReference[] references,
                              int[][] byClass, FullHandModel ml, FullHandModel hybrid,
                              bool synthetic, string digest, string exporterVersion)
        {
            this.vocabulary = vocabulary;
            this.references = references;
            this.referencesByClass = byClass;
            MlModel = ml; HybridModel = hybrid;
            Synthetic = synthetic; PayloadSha256 = digest; ExporterVersion = exporterVersion;
        }

        public string[] CopyVocabulary() { return (string[])vocabulary.Clone(); }
        public string Label(int index) { return vocabulary[index]; }
        public FullHandReference Reference(int index) { return references[index]; }
        public FullHandFeatureSequence ReferenceSequence(int index)
        {
            return references[index].Sequence;
        }
        public int[] ReferencesForClass(int labelIndex)
        {
            return (int[])referencesByClass[labelIndex].Clone();
        }
        public string[] CopyReferenceIds()
        {
            var ids = new string[references.Length];
            for (int index = 0; index < references.Length; index++) ids[index] = references[index].Id;
            return ids;
        }

        static void Fail(string code, string message)
        {
            throw new FullHandFormatException(code, message);
        }

        /// <summary>Loads from an explicitly supplied manifest path. Nothing is searched.</summary>
        public static FullHandRuntimeBundle Load(string manifestPath)
        {
            if (string.IsNullOrEmpty(manifestPath)) Fail(FullHandBundleCodes.ManifestMissing, "no path");
            if (!File.Exists(manifestPath))
                Fail(FullHandBundleCodes.ManifestMissing, Path.GetFileName(manifestPath));
            FullHandBundleManifest manifest = ReadManifest(manifestPath);
            string name = manifest.payload_file;
            if (string.IsNullOrEmpty(name) || name.IndexOf('/') >= 0 || name.IndexOf('\\') >= 0
                || name == "." || name == "..")
                Fail(FullHandBundleCodes.PayloadPathInvalid,
                     "payload_file must be a plain file name beside the manifest");
            string directory = Path.GetDirectoryName(Path.GetFullPath(manifestPath));
            return Load(manifestPath, Path.Combine(directory, name));
        }

        public static FullHandRuntimeBundle Load(string manifestPath, string payloadPath)
        {
            FullHandBundleManifest manifest = ReadManifest(manifestPath);
            if (!File.Exists(payloadPath))
                Fail(FullHandBundleCodes.PayloadMissing, Path.GetFileName(payloadPath));
            var info = new FileInfo(payloadPath);
            if (info.Length > MaxPayloadBytes)
                Fail(FullHandBundleCodes.SizeMismatch, "payload exceeds the documented bound");
            if (info.Length != manifest.payload_bytes)
                Fail(FullHandBundleCodes.SizeMismatch,
                     info.Length + " on disk, " + manifest.payload_bytes + " declared");
            byte[] payload = File.ReadAllBytes(payloadPath);
            if (!string.Equals(Sha256Hex(payload), manifest.payload_sha256,
                               StringComparison.OrdinalIgnoreCase))
                Fail(FullHandBundleCodes.DigestMismatch, "the payload bytes changed");
            return Parse(manifest, payload);
        }

        static FullHandBundleManifest ReadManifest(string manifestPath)
        {
            if (!File.Exists(manifestPath))
                Fail(FullHandBundleCodes.ManifestMissing, Path.GetFileName(manifestPath));
            FullHandBundleManifest manifest = null;
            try
            {
                manifest = JsonUtility.FromJson<FullHandBundleManifest>(
                    File.ReadAllText(manifestPath, new UTF8Encoding(false)));
            }
            catch (Exception error)
            {
                Fail(FullHandBundleCodes.ManifestMalformed, error.GetType().Name);
            }
            if (manifest == null) Fail(FullHandBundleCodes.ManifestMalformed, "empty manifest");
            bool phase = string.Equals(manifest.schema, PhaseSchema, StringComparison.Ordinal);
            if (!phase && !string.Equals(manifest.schema, SupportedSchema,
                                         StringComparison.Ordinal))
                Fail(FullHandBundleCodes.SchemaMismatch, manifest.schema ?? "absent");
            if (manifest.format_version != (phase ? PhaseFormatVersion : SupportedFormatVersion))
                Fail(FullHandBundleCodes.SchemaMismatch,
                     "format version " + manifest.format_version);
            if (!string.Equals(manifest.point_schema, FullHandPairConstants.PairSchemaId,
                               StringComparison.Ordinal))
                Fail(FullHandBundleCodes.SchemaMismatch, manifest.point_schema ?? "absent");
            int expectedFixed = phase ? FullHandPhasePairFeatures.FeatureCount
                                      : FullHandPairConstants.FixedFeatureCount;
            int expectedAligned = phase ? FullHandPhasePairFeatures.FeatureCount
                                        : FullHandPairConstants.AlignedFeatureCount;
            if (manifest.resample_frames != FullHandPairConstants.ResampleFrames ||
                manifest.feature_point_count != FullHandPairConstants.FeaturePointCount ||
                manifest.coord_dims != FullHandPairConstants.CoordDims ||
                manifest.fixed_feature_count != expectedFixed ||
                manifest.aligned_feature_count != expectedAligned)
                Fail(FullHandBundleCodes.DimensionMismatch, "manifest dimensions");
            if (string.IsNullOrEmpty(manifest.payload_sha256) ||
                manifest.payload_sha256.Length != 64)
                Fail(FullHandBundleCodes.ManifestMalformed, "payload_sha256");
            if (manifest.payload_bytes <= 0)
                Fail(FullHandBundleCodes.ManifestMalformed, "payload_bytes");
            return manifest;
        }

        static FullHandRuntimeBundle Parse(FullHandBundleManifest manifest, byte[] payload)
        {
            // The manifest schema decides which payload the bytes must be; the payload magic,
            // version and widths must then agree with it.
            bool phase = string.Equals(manifest.schema, PhaseSchema, StringComparison.Ordinal);
            byte[] magic = phase ? PhaseMagic : Magic;
            int expectedFixed = phase ? FullHandPhasePairFeatures.FeatureCount
                                      : FullHandPairConstants.FixedFeatureCount;
            int expectedAligned = phase ? FullHandPhasePairFeatures.FeatureCount
                                        : FullHandPairConstants.AlignedFeatureCount;
            int cursor = 0;
            for (int index = 0; index < magic.Length; index++)
                if (Need(payload, cursor + index, 1) || payload[cursor + index] != magic[index])
                    Fail(FullHandBundleCodes.HeaderInvalid, "magic");
            cursor += magic.Length;
            if (ReadInt32(payload, ref cursor) !=
                (phase ? PhaseFormatVersion : SupportedFormatVersion))
                Fail(FullHandBundleCodes.HeaderInvalid, "format version");
            if (ReadInt32(payload, ref cursor) != FullHandPairConstants.ResampleFrames ||
                ReadInt32(payload, ref cursor) != FullHandPairConstants.FeaturePointCount ||
                ReadInt32(payload, ref cursor) != FullHandPairConstants.CoordDims ||
                ReadInt32(payload, ref cursor) != expectedFixed ||
                ReadInt32(payload, ref cursor) != expectedAligned)
                Fail(FullHandBundleCodes.DimensionMismatch, "payload dimensions");

            int vocabularyCount = ReadInt32(payload, ref cursor);
            if (vocabularyCount < 1 || vocabularyCount > MaxVocabulary)
                Fail(FullHandBundleCodes.CountInvalid, "vocabulary " + vocabularyCount);
            var vocabulary = new string[vocabularyCount];
            for (int index = 0; index < vocabularyCount; index++)
            {
                vocabulary[index] = ReadString(payload, ref cursor);
                if (vocabulary[index].Length == 0)
                    Fail(FullHandBundleCodes.LabelInvalid, "an empty label");
                if (index > 0 && string.CompareOrdinal(vocabulary[index - 1],
                                                       vocabulary[index]) >= 0)
                    Fail(FullHandBundleCodes.LabelInvalid,
                         "labels must be unique and ordinally ascending");
            }

            FullHandModel ml = phase
                ? ReadPhaseModel(payload, ref cursor)
                : ReadModel(payload, ref cursor, FullHandPairConstants.FixedFeatureCount);
            FullHandModel hybrid = phase
                ? ReadPhaseModel(payload, ref cursor)
                : ReadModel(payload, ref cursor, FullHandPairConstants.AlignedFeatureCount);

            int referenceCount = ReadInt32(payload, ref cursor);
            if (referenceCount < 1 || referenceCount > MaxReferences)
                Fail(FullHandBundleCodes.CountInvalid, "references " + referenceCount);
            int frames = FullHandPairConstants.ResampleFrames;
            int points = FullHandPairConstants.FeaturePointCount;
            int dims = FullHandPairConstants.CoordDims;
            var references = new FullHandReference[referenceCount];
            var seen = new HashSet<string>(StringComparer.Ordinal);
            var byClass = new List<int>[vocabularyCount];
            for (int index = 0; index < vocabularyCount; index++) byClass[index] = new List<int>();

            for (int index = 0; index < referenceCount; index++)
            {
                string id = ReadString(payload, ref cursor);
                if (id.Length == 0 || !seen.Add(id))
                    Fail(FullHandBundleCodes.ReferenceInvalid, "an empty or duplicate id");
                int labelIndex = ReadInt32(payload, ref cursor);
                if (labelIndex < 0 || labelIndex >= vocabularyCount)
                    Fail(FullHandBundleCodes.ReferenceInvalid, "unknown label index");
                var coordinates = new double[frames * points * dims];
                for (int slot = 0; slot < coordinates.Length; slot++)
                    coordinates[slot] = ReadDouble(payload, ref cursor);
                var mask = new bool[frames * points];
                for (int slot = 0; slot < mask.Length; slot++)
                {
                    if (Need(payload, cursor, 1)) Fail(FullHandBundleCodes.Truncated, "mask");
                    byte raw = payload[cursor++];
                    if (raw > 1) Fail(FullHandBundleCodes.ReferenceInvalid, "mask value");
                    mask[slot] = raw == 1;
                }
                bool[] available = ValidateReference(coordinates, mask);
                references[index] = new FullHandReference(id, labelIndex,
                    new FullHandFeatureSequence(coordinates, mask, available,
                                                double.NaN, double.NaN, frames, true, null));
                byClass[labelIndex].Add(index);
            }
            if (cursor != payload.Length)
                Fail(FullHandBundleCodes.TrailingBytes,
                     (payload.Length - cursor) + " unread byte(s)");

            CrossCheck(manifest, vocabulary, references);
            var resolved = new int[vocabularyCount][];
            for (int index = 0; index < vocabularyCount; index++)
                resolved[index] = byClass[index].ToArray();
            return new FullHandRuntimeBundle(vocabulary, references, resolved, ml, hybrid,
                                             manifest.synthetic, manifest.payload_sha256,
                                             manifest.exporter_version ?? string.Empty);
        }

        static bool[] ValidateReference(double[] coordinates, bool[] mask)
        {
            int points = FullHandPairConstants.FeaturePointCount;
            int dims = FullHandPairConstants.CoordDims;
            var available = new bool[FullHandPairConstants.GroupCount];
            for (int slot = 0; slot < mask.Length; slot++)
            {
                double x = coordinates[slot * dims];
                double y = coordinates[slot * dims + 1];
                double z = coordinates[slot * dims + 2];
                if (!mask[slot])
                {
                    if (!double.IsNaN(x) || !double.IsNaN(y) || !double.IsNaN(z))
                        Fail(FullHandBundleCodes.ReferenceInvalid,
                             "a masked slot must hold NaN in all three channels");
                    continue;
                }
                if (double.IsNaN(x) || double.IsInfinity(x) ||
                    double.IsNaN(y) || double.IsInfinity(y))
                    Fail(FullHandBundleCodes.ReferenceInvalid, "a valid point is not finite");
                if (z != 0.0)
                    Fail(FullHandBundleCodes.ReferenceInvalid, "a valid point has a non-zero z");
                int point = slot % points;
                for (int group = 0; group < FullHandPairConstants.GroupCount; group++)
                    if (point >= FullHandPairConstants.GroupStart[group] &&
                        point < FullHandPairConstants.GroupEnd[group])
                        available[group] = true;
            }
            if (!available[3] && !available[4])
                Fail(FullHandBundleCodes.ReferenceInvalid,
                     "a reference carries no local hand shape and could never be usable");
            return available;
        }

        static void CrossCheck(FullHandBundleManifest manifest, string[] vocabulary,
                               FullHandReference[] references)
        {
            if (manifest.vocabulary_count != vocabulary.Length ||
                manifest.reference_count != references.Length)
                Fail(FullHandBundleCodes.ManifestPayloadDisagree, "declared counts");
            if (manifest.vocabulary == null || manifest.vocabulary.Length != vocabulary.Length)
                Fail(FullHandBundleCodes.ManifestPayloadDisagree, "vocabulary length");
            for (int index = 0; index < vocabulary.Length; index++)
                if (!string.Equals(manifest.vocabulary[index], vocabulary[index],
                                   StringComparison.Ordinal))
                    Fail(FullHandBundleCodes.ManifestPayloadDisagree, "vocabulary entry");
            if (manifest.reference_ids == null ||
                manifest.reference_ids.Length != references.Length ||
                manifest.reference_label_indices == null ||
                manifest.reference_label_indices.Length != references.Length)
                Fail(FullHandBundleCodes.ManifestPayloadDisagree, "reference tables");
            for (int index = 0; index < references.Length; index++)
            {
                if (!string.Equals(manifest.reference_ids[index], references[index].Id,
                                   StringComparison.Ordinal))
                    Fail(FullHandBundleCodes.ManifestPayloadDisagree, "reference id");
                if (manifest.reference_label_indices[index] != references[index].LabelIndex)
                    Fail(FullHandBundleCodes.ManifestPayloadDisagree, "reference label");
            }
        }

        /// <summary>
        /// Phase model: the classic layout at width 432, then double[432] imputation means
        /// straight after the intercept. Nothing else about the payload layout changes.
        /// </summary>
        static FullHandModel ReadPhaseModel(byte[] payload, ref int cursor)
        {
            int width = ReadInt32(payload, ref cursor);
            if (width != FullHandPhasePairFeatures.FeatureCount)
                Fail(FullHandBundleCodes.ModelInvalid,
                     "phase model width " + width + ", expected "
                     + FullHandPhasePairFeatures.FeatureCount);
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            var imputation = new double[width];
            for (int index = 0; index < width; index++) mean[index] = ReadDouble(payload, ref cursor);
            for (int index = 0; index < width; index++) scale[index] = ReadDouble(payload, ref cursor);
            for (int index = 0; index < width; index++)
                coefficients[index] = ReadDouble(payload, ref cursor);
            double intercept = ReadDouble(payload, ref cursor);
            for (int index = 0; index < width; index++)
                imputation[index] = ReadDouble(payload, ref cursor);
            try
            {
                return new FullHandModel(mean, scale, coefficients, intercept, imputation);
            }
            catch (FullHandFormatException error)
            {
                Fail(FullHandBundleCodes.ModelInvalid, error.Code);
                return null;
            }
        }

        static FullHandModel ReadModel(byte[] payload, ref int cursor, int expectedWidth)
        {
            int width = ReadInt32(payload, ref cursor);
            if (width != expectedWidth)
                Fail(FullHandBundleCodes.ModelInvalid,
                     "model width " + width + ", expected " + expectedWidth);
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            for (int index = 0; index < width; index++) mean[index] = ReadDouble(payload, ref cursor);
            for (int index = 0; index < width; index++) scale[index] = ReadDouble(payload, ref cursor);
            for (int index = 0; index < width; index++)
                coefficients[index] = ReadDouble(payload, ref cursor);
            double intercept = ReadDouble(payload, ref cursor);
            try
            {
                return new FullHandModel(mean, scale, coefficients, intercept);
            }
            catch (FullHandFormatException error)
            {
                Fail(FullHandBundleCodes.ModelInvalid, error.Code);
                return null;
            }
        }

        static bool Need(byte[] payload, int cursor, int count)
        {
            return cursor < 0 || (long)cursor + count > payload.Length;
        }

        static int ReadInt32(byte[] payload, ref int cursor)
        {
            if (Need(payload, cursor, 4)) Fail(FullHandBundleCodes.Truncated, "int32");
            int value = payload[cursor] | (payload[cursor + 1] << 8)
                      | (payload[cursor + 2] << 16) | (payload[cursor + 3] << 24);
            cursor += 4;
            return value;
        }

        static double ReadDouble(byte[] payload, ref int cursor)
        {
            if (Need(payload, cursor, 8)) Fail(FullHandBundleCodes.Truncated, "double");
            double value;
            if (BitConverter.IsLittleEndian) value = BitConverter.ToDouble(payload, cursor);
            else
            {
                var swapped = new byte[8];
                for (int index = 0; index < 8; index++) swapped[index] = payload[cursor + 7 - index];
                value = BitConverter.ToDouble(swapped, 0);
            }
            cursor += 8;
            return value;
        }

        static string ReadString(byte[] payload, ref int cursor)
        {
            int length = ReadInt32(payload, ref cursor);
            if (length < 0 || length > MaxStringBytes)
                Fail(FullHandBundleCodes.CountInvalid, "string length " + length);
            if (Need(payload, cursor, length)) Fail(FullHandBundleCodes.Truncated, "string");
            string value = new UTF8Encoding(false).GetString(payload, cursor, length);
            cursor += length;
            return value;
        }

        static string Sha256Hex(byte[] data)
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
}
