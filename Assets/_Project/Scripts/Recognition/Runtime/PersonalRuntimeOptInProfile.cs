using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Recognition.Runtime
{
    public enum PersonalProfileFailureCode
    {
        ProfileDirectoryMissing,
        ManifestMissing,
        ManifestMalformed,
        ManifestSchemaMismatch,
        UnsafeRelativePath,
        AssetMissing,
        AssetSizeMismatch,
        AssetSha256Mismatch,
        ConfigMalformed,
        ConfigSchemaMismatch,
        DuplicateWord,
        WordSetMismatch,
        ThresholdInvalid,
        WisrtFormatInvalid,
        EmbeddedChecksumMismatch,
        ProvenanceMismatch,
    }

    /// <summary>Typed, fail-closed personal-profile load failure. It is never converted to a global fallback.</summary>
    public sealed class PersonalProfileLoadException : Exception
    {
        public PersonalProfileFailureCode FailureCode { get; }

        public PersonalProfileLoadException(PersonalProfileFailureCode failureCode, string message)
            : base(message)
        {
            FailureCode = failureCode;
        }

        public PersonalProfileLoadException(PersonalProfileFailureCode failureCode, string message, Exception innerException)
            : base(message, innerException)
        {
            FailureCode = failureCode;
        }
    }

    public sealed class PersonalTemplateProvenance
    {
        public string Word { get; }
        public string RelativePath { get; }
        public long SizeBytes { get; }
        public string FileSha256 { get; }
        public string EmbeddedChecksumSha256 { get; }
        public string CaptureId { get; }
        public string CaptureKey { get; }
        public string RunId { get; }
        public string SourcePayloadSha256 { get; }
        public string CoordinatePayloadSha256 { get; }
        public string MaskPayloadSha256 { get; }
        public string NormalizedPayloadSha256 { get; }

        internal PersonalTemplateProvenance(
            string word,
            string relativePath,
            long sizeBytes,
            string fileSha256,
            string embeddedChecksumSha256,
            string captureId,
            string captureKey,
            string runId,
            string sourcePayloadSha256,
            string coordinatePayloadSha256,
            string maskPayloadSha256,
            string normalizedPayloadSha256)
        {
            Word = word;
            RelativePath = relativePath;
            SizeBytes = sizeBytes;
            FileSha256 = fileSha256;
            EmbeddedChecksumSha256 = embeddedChecksumSha256;
            CaptureId = captureId;
            CaptureKey = captureKey;
            RunId = runId;
            SourcePayloadSha256 = sourcePayloadSha256;
            CoordinatePayloadSha256 = coordinatePayloadSha256;
            MaskPayloadSha256 = maskPayloadSha256;
            NormalizedPayloadSha256 = normalizedPayloadSha256;
        }
    }

    /// <summary>Fully validated immutable Step 9H personal profile. Loading never changes runtime mode.</summary>
    public sealed class PersonalRuntimeProfile
    {
        readonly IReadOnlyDictionary<string, RecognitionSequence> templates;
        readonly IReadOnlyDictionary<string, double> thresholds;
        readonly IReadOnlyDictionary<string, PersonalTemplateProvenance> provenance;

        public string ProfileId { get; }
        public bool OptInOnly => true;
        public bool DefaultActive => false;
        public bool GreetingMarginNarrow => true;
        public IReadOnlyList<string> Words { get; }
        public IReadOnlyDictionary<string, RecognitionSequence> Templates => templates;
        public IReadOnlyDictionary<string, double> Thresholds => thresholds;
        public IReadOnlyDictionary<string, PersonalTemplateProvenance> TemplateProvenance => provenance;
        public IReadOnlyList<string> Warnings { get; }

        internal PersonalRuntimeProfile(
            string profileId,
            IDictionary<string, RecognitionSequence> templates,
            IDictionary<string, double> thresholds,
            IDictionary<string, PersonalTemplateProvenance> provenance,
            IEnumerable<string> warnings)
        {
            ProfileId = profileId;
            this.templates = new ReadOnlyDictionary<string, RecognitionSequence>(
                new Dictionary<string, RecognitionSequence>(templates, StringComparer.Ordinal));
            this.thresholds = new ReadOnlyDictionary<string, double>(
                new Dictionary<string, double>(thresholds, StringComparer.Ordinal));
            this.provenance = new ReadOnlyDictionary<string, PersonalTemplateProvenance>(
                new Dictionary<string, PersonalTemplateProvenance>(provenance, StringComparer.Ordinal));
            Words = Array.AsReadOnly(PersonalRuntimeProfileLoader.RequiredWords.ToArray());
            Warnings = Array.AsReadOnly(warnings.ToArray());
        }
    }

    /// <summary>Strict loader for the installed, default-inactive Step 9H profile.</summary>
    public static class PersonalRuntimeProfileLoader
    {
        public const string ProfileId = "step9h_current_user_v1_0";
        public const string ManifestFileName = "personal_runtime_profile_manifest.json";
        public const string ConfigFileName = "personal_class_threshold_config.json";
        public const string Step9HDeliverySha256 = "4827e81e029e63e7939e7ed91cb33ac5f90dd6822fa13722da01f642aed2ddb9";
        public const long Step9HDeliverySizeBytes = 3214867;
        public const string ConfigFileSha256 = "e84edd1f598c2d94f8412767f3125031e0be330886d3a315ed429798fe49726e";
        public const long ConfigFileSizeBytes = 4811;
        public const string GreetingMarginWarning = "GREETING_CLASS_CONDITIONAL_MARGIN_NARROW";

        internal static readonly string[] RequiredWords = { "GREETING", "LOVE", "THANK_YOU" };

        sealed class ExpectedTemplate
        {
            public string Word;
            public string FileName;
            public long SizeBytes;
            public string FileSha256;
            public string EmbeddedChecksumSha256;
            public string CaptureId;
            public string CaptureKey;
            public string RunId;
            public string SourcePayloadSha256;
            public string CoordinatePayloadSha256;
            public string MaskPayloadSha256;
            public string NormalizedPayloadSha256;
        }

        sealed class ManifestAsset
        {
            public string Kind;
            public string Word;
            public string RelativePath;
            public long SizeBytes;
            public string Sha256;
            public string EmbeddedChecksumSha256;
            public string CaptureId;
            public string CaptureKey;
            public string RunId;
            public string SourcePayloadSha256;
            public string CoordinatePayloadSha256;
            public string MaskPayloadSha256;
            public string NormalizedPayloadSha256;
        }

        static readonly ExpectedTemplate[] ExpectedTemplates =
        {
            new ExpectedTemplate
            {
                Word = "GREETING",
                FileName = "personal_template_GREETING.wisrt",
                SizeBytes = 53564,
                FileSha256 = "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681",
                EmbeddedChecksumSha256 = "bd9c9c421f51df4e89ae9119427ac555e513a025ecdc6ec48e1d7858e96e3a26",
                CaptureId = "87e5b54432ac49a197a596d5ff7c3845",
                CaptureKey = "step9f_additional::87e5b54432ac49a197a596d5ff7c3845",
                RunId = "3698e6b5a88d438d9f94b1742978eb36",
                SourcePayloadSha256 = "146897a4769bb164b9fdb013a6897437462a423c1fe9ef15cb6ce9d4cdbcc240",
                CoordinatePayloadSha256 = "0b6c1ac2cfc4422af014a79f49acfdf9bfe53bf36a1c08c80c336fd978d3618e",
                MaskPayloadSha256 = "d8e56cd267758595645f298a3dd3ad665943ced8a583918754a9add21c4f731d",
                NormalizedPayloadSha256 = "911dc3db1f079ed0010dbfe3a431c32f6bbbd7fc77c1803b05f8a1bc03c9d975",
            },
            new ExpectedTemplate
            {
                Word = "LOVE",
                FileName = "personal_template_LOVE.wisrt",
                SizeBytes = 56262,
                FileSha256 = "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b",
                EmbeddedChecksumSha256 = "d88398c029222013da4be539172a47f8dbb9ca2e39645b937518bc6bfd6d98e8",
                CaptureId = "d958cbc2a1c44e03bb29bf64a6fae195",
                CaptureKey = "step9f_additional::d958cbc2a1c44e03bb29bf64a6fae195",
                RunId = "3698e6b5a88d438d9f94b1742978eb36",
                SourcePayloadSha256 = "cd9f516cf4ae1575a6334577d5328dc843b93d330bca79f81d55ef03ba8393ad",
                CoordinatePayloadSha256 = "c6731402913b61b66eaa47f069e3f8e9f525e01e13756f6834c56be2db7b6ea4",
                MaskPayloadSha256 = "6e613d2fddf0efa3e136be41214504180db94c85c231aa071964f97836c0adc4",
                NormalizedPayloadSha256 = "5478fadab109b02faa260b7c2f3e58560968f81d773b98a00883a922f02648dc",
            },
            new ExpectedTemplate
            {
                Word = "THANK_YOU",
                FileName = "personal_template_THANK_YOU.wisrt",
                SizeBytes = 49319,
                FileSha256 = "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5",
                EmbeddedChecksumSha256 = "1b05c1c912916bd0ed90ab43f50717c177a6b83a0a482cdfeafbb262a7f46e00",
                CaptureId = "2b323671e3f64049a74556c71ff70154",
                CaptureKey = "raw_intermediate_step9d::2b323671e3f64049a74556c71ff70154",
                RunId = "e965f94530f6432c800eea60be8f2cdf",
                SourcePayloadSha256 = "858042fa4b3474b366c0236e0a8693f366204530366656f18fff74a55289dfc9",
                CoordinatePayloadSha256 = "490dbeb1e50953dd0d51ad2de570ae2534f786af833fcadf0857d2e005a410d9",
                MaskPayloadSha256 = "919b9da5efa1ea9fd1bd17a8bcd8412d1c27bd2c790ef091e8ecd240ae85a113",
                NormalizedPayloadSha256 = "6223c805af9245254ee223ba832eaa916d85e9d2d923ff81e161f34503c63644",
            },
        };

        static readonly IReadOnlyDictionary<string, double> ExpectedThresholds =
            new ReadOnlyDictionary<string, double>(new Dictionary<string, double>(StringComparer.Ordinal)
            {
                { "GREETING", 0.5205639017143886 },
                { "LOVE", 0.39564149928871895 },
                { "THANK_YOU", 0.38124865401356706 },
            });

        public static PersonalRuntimeProfile Load(string profileDirectory)
        {
            if (string.IsNullOrWhiteSpace(profileDirectory))
                throw Failure(PersonalProfileFailureCode.ProfileDirectoryMissing, "personal profile directory is required");
            if (!Directory.Exists(profileDirectory))
                throw Failure(PersonalProfileFailureCode.ProfileDirectoryMissing, "personal profile directory is missing");

            string manifestPath = Path.Combine(profileDirectory, ManifestFileName);
            if (!File.Exists(manifestPath))
                throw Failure(PersonalProfileFailureCode.ManifestMissing, "personal profile manifest is missing");

            Dictionary<string, object> manifest;
            try
            {
                manifest = StrictJson.ParseObject(File.ReadAllBytes(manifestPath));
            }
            catch (Exception exc)
            {
                throw Failure(PersonalProfileFailureCode.ManifestMalformed, "personal profile manifest is malformed", exc);
            }

            try
            {
                ValidateManifestHeader(manifest);
                List<ManifestAsset> manifestAssets = ReadManifestAssets(manifest);
                ManifestAsset configAsset = manifestAssets.Single(item => item.Kind == "threshold_config");
                Dictionary<string, ManifestAsset> templateAssets = manifestAssets
                    .Where(item => item.Kind == "template")
                    .ToDictionary(item => item.Word, StringComparer.Ordinal);

                string configPath = ResolveSafe(profileDirectory, configAsset.RelativePath);
                byte[] configBytes = ReadAsset(configPath, configAsset);
                Dictionary<string, double> thresholds;
                Dictionary<string, Dictionary<string, object>> configTemplates;
                ValidateConfig(configBytes, out thresholds, out configTemplates);
                ValidateAssetSha(configBytes, configAsset, ConfigFileSha256);

                var templates = new Dictionary<string, RecognitionSequence>(StringComparer.Ordinal);
                var provenance = new Dictionary<string, PersonalTemplateProvenance>(StringComparer.Ordinal);
                foreach (ExpectedTemplate expected in ExpectedTemplates)
                {
                    ManifestAsset asset;
                    if (!templateAssets.TryGetValue(expected.Word, out asset))
                        throw Failure(PersonalProfileFailureCode.WordSetMismatch, "personal template word is missing: " + expected.Word);
                    string assetPath = ResolveSafe(profileDirectory, asset.RelativePath);
                    ValidateManifestTemplate(asset, expected);
                    byte[] bytes = ReadAsset(assetPath, asset);
                    RecognitionSequence sequence;
                    try
                    {
                        sequence = RecognitionSequenceBinary.Load(bytes);
                    }
                    catch (InvalidDataException exc)
                    {
                        PersonalProfileFailureCode code = exc.Message.IndexOf("payload SHA-256", StringComparison.Ordinal) >= 0
                            ? PersonalProfileFailureCode.EmbeddedChecksumMismatch
                            : PersonalProfileFailureCode.WisrtFormatInvalid;
                        throw Failure(code, "personal WISRT validation failed for " + expected.Word, exc);
                    }
                    ValidateAssetSha(bytes, asset, expected.FileSha256);
                    ValidateSequence(sequence, expected, configTemplates[expected.Word]);
                    templates.Add(expected.Word, sequence);
                    provenance.Add(expected.Word, new PersonalTemplateProvenance(
                        expected.Word, asset.RelativePath, asset.SizeBytes, asset.Sha256,
                        asset.EmbeddedChecksumSha256, asset.CaptureId, asset.CaptureKey, asset.RunId,
                        asset.SourcePayloadSha256, asset.CoordinatePayloadSha256,
                        asset.MaskPayloadSha256, asset.NormalizedPayloadSha256));
                }

                if (Directory.GetFiles(profileDirectory, "*.wisrt", SearchOption.TopDirectoryOnly).Length != 3)
                    throw Failure(PersonalProfileFailureCode.WordSetMismatch, "personal profile must contain exactly three WISRT files");

                return new PersonalRuntimeProfile(
                    ProfileId, templates, thresholds, provenance, new[] { GreetingMarginWarning });
            }
            catch (PersonalProfileLoadException)
            {
                throw;
            }
            catch (Exception exc)
            {
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "personal profile validation failed", exc);
            }
        }

        static void ValidateManifestHeader(Dictionary<string, object> manifest)
        {
            ExactKeys(manifest,
                "schema_version", "profile_id", "mode", "threshold_mode", "opt_in_only",
                "default_active", "installed_but_default_inactive", "template_count", "threshold_count",
                "words", "source_step9h_delivery", "assets", "global_assets_overwritten", "warnings");
            RequireEqual(StringValue(manifest, "schema_version"),
                "words_in_silence.step9i.personal_runtime_profile.v1_0", "manifest schema mismatch");
            RequireEqual(StringValue(manifest, "profile_id"), ProfileId, "profile ID mismatch");
            RequireEqual(StringValue(manifest, "mode"), "personal", "profile mode mismatch");
            RequireEqual(StringValue(manifest, "threshold_mode"), "class_conditional_one_vs_rest", "threshold mode mismatch");
            if (!BoolValue(manifest, "opt_in_only") || BoolValue(manifest, "default_active") ||
                !BoolValue(manifest, "installed_but_default_inactive") || BoolValue(manifest, "global_assets_overwritten"))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "personal profile activation/global flags are invalid");
            if (Int64Value(manifest, "template_count") != 3 || Int64Value(manifest, "threshold_count") != 3)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "personal profile counts are invalid");

            string[] words = StringArray(manifest, "words");
            ValidateWords(words);
            string[] warnings = StringArray(manifest, "warnings");
            if (warnings.Length != 1 || warnings[0] != GreetingMarginWarning)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "personal profile warning set is invalid");

            Dictionary<string, object> source = ObjectValue(manifest, "source_step9h_delivery");
            ExactKeys(source, "relative_path", "size_bytes", "sha256");
            RequireEqual(StringValue(source, "relative_path"),
                "Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip",
                "Step 9H source path mismatch");
            RequireEqual(StringValue(source, "sha256"), Step9HDeliverySha256, "Step 9H source SHA-256 mismatch");
            if (Int64Value(source, "size_bytes") != Step9HDeliverySizeBytes)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "Step 9H source size mismatch");
        }

        static List<ManifestAsset> ReadManifestAssets(Dictionary<string, object> manifest)
        {
            List<object> raw = ArrayValue(manifest, "assets");
            if (raw.Count != 4)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "exactly four personal profile assets are required");
            var result = new List<ManifestAsset>();
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (object value in raw)
            {
                var record = value as Dictionary<string, object>;
                if (record == null)
                    throw Failure(PersonalProfileFailureCode.ManifestMalformed, "personal asset record is not an object");
                string kind = StringValue(record, "kind");
                if (kind == "template")
                {
                    ExactKeys(record, "kind", "word", "relative_path", "size_bytes", "sha256",
                        "embedded_checksum_sha256", "capture_id", "capture_key", "run_id",
                        "source_payload_sha256", "coordinate_payload_sha256", "mask_payload_sha256",
                        "normalized_payload_sha256");
                    var asset = new ManifestAsset
                    {
                        Kind = kind,
                        Word = StringValue(record, "word"),
                        RelativePath = StringValue(record, "relative_path"),
                        SizeBytes = Int64Value(record, "size_bytes"),
                        Sha256 = StringValue(record, "sha256"),
                        EmbeddedChecksumSha256 = StringValue(record, "embedded_checksum_sha256"),
                        CaptureId = StringValue(record, "capture_id"),
                        CaptureKey = StringValue(record, "capture_key"),
                        RunId = StringValue(record, "run_id"),
                        SourcePayloadSha256 = StringValue(record, "source_payload_sha256"),
                        CoordinatePayloadSha256 = StringValue(record, "coordinate_payload_sha256"),
                        MaskPayloadSha256 = StringValue(record, "mask_payload_sha256"),
                        NormalizedPayloadSha256 = StringValue(record, "normalized_payload_sha256"),
                    };
                    if (!keys.Add("template|" + asset.Word))
                        throw Failure(PersonalProfileFailureCode.DuplicateWord, "duplicate personal template word: " + asset.Word);
                    result.Add(asset);
                }
                else if (kind == "threshold_config")
                {
                    ExactKeys(record, "kind", "relative_path", "size_bytes", "sha256");
                    var asset = new ManifestAsset
                    {
                        Kind = kind,
                        Word = string.Empty,
                        RelativePath = StringValue(record, "relative_path"),
                        SizeBytes = Int64Value(record, "size_bytes"),
                        Sha256 = StringValue(record, "sha256"),
                    };
                    if (!keys.Add("threshold_config"))
                        throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "duplicate threshold config record");
                    result.Add(asset);
                }
                else
                {
                    throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "unsupported personal asset kind: " + kind);
                }
            }
            if (result.Count(item => item.Kind == "template") != 3 || result.Count(item => item.Kind == "threshold_config") != 1)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "personal asset kind counts are invalid");
            ValidateWords(result.Where(item => item.Kind == "template").Select(item => item.Word).ToArray());
            return result;
        }

        static void ValidateConfig(
            byte[] bytes,
            out Dictionary<string, double> thresholds,
            out Dictionary<string, Dictionary<string, object>> templateRecords)
        {
            Dictionary<string, object> config;
            try
            {
                config = StrictJson.ParseObject(bytes);
            }
            catch (Exception exc)
            {
                throw Failure(PersonalProfileFailureCode.ConfigMalformed, "personal threshold config is malformed", exc);
            }

            RequireEqual(StringValue(config, "schema_version"),
                "words_in_silence.step9h.personalized_class_conditional_threshold.v1_0.personal_class_threshold_config",
                "personal threshold config schema mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            RequireEqual(StringValue(config, "mode"), "personal", "personal threshold mode mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            RequireEqual(StringValue(config, "threshold_mode"), "class_conditional_one_vs_rest",
                "personal threshold policy mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            if (!BoolValue(config, "opt_in_only") || BoolValue(config, "runtime_activated") ||
                BoolValue(config, "global_template_or_threshold_overwritten") || !BoolValue(config, "greeting_margin_narrow"))
                throw Failure(PersonalProfileFailureCode.ConfigSchemaMismatch, "personal threshold activation/global flags are invalid");
            if (Int64Value(config, "source_capture_count") != 9)
                throw Failure(PersonalProfileFailureCode.ConfigSchemaMismatch, "personal threshold source capture count is invalid");

            Dictionary<string, object> thresholdObject = ObjectValue(config, "thresholds");
            ExactKeys(thresholdObject, RequiredWords);
            thresholds = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (string word in RequiredWords)
            {
                double threshold = NumberValue(thresholdObject, word);
                if (double.IsNaN(threshold) || double.IsInfinity(threshold) || threshold < 0.0)
                    throw Failure(PersonalProfileFailureCode.ThresholdInvalid, "personal threshold is non-finite or negative: " + word);
                if (threshold != ExpectedThresholds[word])
                    throw Failure(PersonalProfileFailureCode.ThresholdInvalid, "personal threshold differs from approved Step 9H value: " + word);
                thresholds.Add(word, threshold);
            }

            string[] warnings = StringArray(config, "warnings");
            if (warnings.Length != 1 || warnings[0] != GreetingMarginWarning)
                throw Failure(PersonalProfileFailureCode.ConfigSchemaMismatch, "personal threshold warning is missing");

            List<object> rawTemplates = ArrayValue(config, "template_records");
            if (rawTemplates.Count != 3)
                throw Failure(PersonalProfileFailureCode.WordSetMismatch, "personal threshold config must contain three template records");
            templateRecords = new Dictionary<string, Dictionary<string, object>>(StringComparer.Ordinal);
            foreach (object raw in rawTemplates)
            {
                var record = raw as Dictionary<string, object>;
                if (record == null)
                    throw Failure(PersonalProfileFailureCode.ConfigMalformed, "personal template provenance record is not an object");
                string word = StringValue(record, "word");
                if (templateRecords.ContainsKey(word))
                    throw Failure(PersonalProfileFailureCode.DuplicateWord, "duplicate config template word: " + word);
                templateRecords.Add(word, record);
            }
            ValidateWords(templateRecords.Keys.ToArray());
        }

        static void ValidateManifestTemplate(ManifestAsset asset, ExpectedTemplate expected)
        {
            if (asset.RelativePath != expected.FileName || asset.SizeBytes != expected.SizeBytes ||
                asset.Sha256 != expected.FileSha256 || asset.EmbeddedChecksumSha256 != expected.EmbeddedChecksumSha256 ||
                asset.CaptureId != expected.CaptureId || asset.CaptureKey != expected.CaptureKey || asset.RunId != expected.RunId ||
                asset.SourcePayloadSha256 != expected.SourcePayloadSha256 ||
                asset.CoordinatePayloadSha256 != expected.CoordinatePayloadSha256 ||
                asset.MaskPayloadSha256 != expected.MaskPayloadSha256 ||
                asset.NormalizedPayloadSha256 != expected.NormalizedPayloadSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "personal manifest provenance mismatch: " + expected.Word);
        }

        static void ValidateSequence(
            RecognitionSequence sequence,
            ExpectedTemplate expected,
            Dictionary<string, object> configRecord)
        {
            if (sequence.SampleId != expected.CaptureId || sequence.Word != expected.Word ||
                sequence.Speed != "PERSONAL" || sequence.Split != "personal_calibration" ||
                sequence.SourceNpzSha256 != expected.SourcePayloadSha256 ||
                sequence.LogicalPayloadSha256 != expected.NormalizedPayloadSha256 ||
                sequence.BinaryPayloadSha256 != expected.EmbeddedChecksumSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "personal WISRT metadata mismatch: " + expected.Word);
            if (Sha256Coordinates(sequence) != expected.CoordinatePayloadSha256 ||
                Sha256Mask(sequence) != expected.MaskPayloadSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "personal WISRT coordinate/mask provenance mismatch: " + expected.Word);

            if (StringValue(configRecord, "capture_id") != expected.CaptureId ||
                StringValue(configRecord, "capture_key") != expected.CaptureKey ||
                StringValue(configRecord, "run_id") != expected.RunId ||
                StringValue(configRecord, "relative_path") != "active_personal_templates/" + expected.FileName ||
                StringValue(configRecord, "sha256") != expected.FileSha256 ||
                Int64Value(configRecord, "size_bytes") != expected.SizeBytes ||
                StringValue(configRecord, "embedded_checksum_sha256") != expected.EmbeddedChecksumSha256 ||
                StringValue(configRecord, "source_payload_sha256") != expected.SourcePayloadSha256 ||
                StringValue(configRecord, "coordinate_payload_sha256") != expected.CoordinatePayloadSha256 ||
                StringValue(configRecord, "mask_payload_sha256") != expected.MaskPayloadSha256 ||
                StringValue(configRecord, "normalized_payload_sha256") != expected.NormalizedPayloadSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "config/template provenance mismatch: " + expected.Word);
        }

        static byte[] ReadAsset(string path, ManifestAsset asset)
        {
            if (!File.Exists(path))
                throw Failure(PersonalProfileFailureCode.AssetMissing, "personal profile asset is missing: " + asset.RelativePath);
            var info = new FileInfo(path);
            if (info.Length != asset.SizeBytes)
                throw Failure(PersonalProfileFailureCode.AssetSizeMismatch, "personal profile asset size mismatch: " + asset.RelativePath);
            return File.ReadAllBytes(path);
        }

        static void ValidateAssetSha(byte[] bytes, ManifestAsset asset, string expectedApprovedSha)
        {
            string actual = Sha256(bytes);
            if (actual != asset.Sha256 || actual != expectedApprovedSha)
                throw Failure(PersonalProfileFailureCode.AssetSha256Mismatch, "personal profile asset SHA-256 mismatch: " + asset.RelativePath);
        }

        static string ResolveSafe(string root, string relativePath)
        {
            if (string.IsNullOrWhiteSpace(relativePath) || relativePath.IndexOf('\\') >= 0 ||
                Path.IsPathRooted(relativePath) || relativePath.IndexOf(':') >= 0)
                throw Failure(PersonalProfileFailureCode.UnsafeRelativePath, "unsafe personal profile relative path");
            string[] segments = relativePath.Split('/');
            if (segments.Any(segment => segment.Length == 0 || segment == "." || segment == ".."))
                throw Failure(PersonalProfileFailureCode.UnsafeRelativePath, "unsafe personal profile relative path");
            string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            string fullPath = Path.GetFullPath(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)));
            if (!fullPath.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase))
                throw Failure(PersonalProfileFailureCode.UnsafeRelativePath, "personal profile relative path escapes its root");
            return fullPath;
        }

        static void ValidateWords(string[] words)
        {
            if (words.Length != RequiredWords.Length)
                throw Failure(PersonalProfileFailureCode.WordSetMismatch, "personal profile word count mismatch");
            if (words.Distinct(StringComparer.Ordinal).Count() != words.Length)
                throw Failure(PersonalProfileFailureCode.DuplicateWord, "personal profile contains a duplicate word");
            string[] sorted = words.OrderBy(item => item, StringComparer.Ordinal).ToArray();
            if (!sorted.SequenceEqual(RequiredWords, StringComparer.Ordinal))
                throw Failure(PersonalProfileFailureCode.WordSetMismatch, "personal profile word set mismatch");
        }

        static string Sha256Coordinates(RecognitionSequence sequence)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                foreach (double value in sequence.CopyCoordinates()) writer.Write(value);
                writer.Flush();
                return Sha256(stream.ToArray());
            }
        }

        static string Sha256Mask(RecognitionSequence sequence)
        {
            bool[] mask = sequence.CopyValidMask();
            var bytes = new byte[mask.Length];
            for (int index = 0; index < mask.Length; index++) bytes[index] = mask[index] ? (byte)1 : (byte)0;
            return Sha256(bytes);
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }

        static PersonalProfileLoadException Failure(PersonalProfileFailureCode code, string message, Exception inner = null)
        {
            return inner == null
                ? new PersonalProfileLoadException(code, message)
                : new PersonalProfileLoadException(code, message, inner);
        }

        static void RequireEqual(string actual, string expected, string message,
            PersonalProfileFailureCode code = PersonalProfileFailureCode.ManifestSchemaMismatch)
        {
            if (actual != expected) throw Failure(code, message);
        }

        static void ExactKeys(Dictionary<string, object> value, params string[] expected)
        {
            var actual = new HashSet<string>(value.Keys, StringComparer.Ordinal);
            var required = new HashSet<string>(expected, StringComparer.Ordinal);
            if (!actual.SetEquals(required))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch,
                    "JSON object key set mismatch; actual=" + string.Join(",", actual.OrderBy(item => item, StringComparer.Ordinal)));
        }

        static object Required(Dictionary<string, object> value, string name)
        {
            object result;
            if (!value.TryGetValue(name, out result))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "required JSON field is missing: " + name);
            return result;
        }

        static string StringValue(Dictionary<string, object> value, string name)
        {
            string result = Required(value, name) as string;
            if (string.IsNullOrWhiteSpace(result))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON string field is invalid: " + name);
            return result;
        }

        static bool BoolValue(Dictionary<string, object> value, string name)
        {
            object raw = Required(value, name);
            if (!(raw is bool))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON Boolean field is invalid: " + name);
            return (bool)raw;
        }

        static double NumberValue(Dictionary<string, object> value, string name)
        {
            object raw = Required(value, name);
            if (!(raw is double))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON numeric field is invalid: " + name);
            return (double)raw;
        }

        static long Int64Value(Dictionary<string, object> value, string name)
        {
            double raw = NumberValue(value, name);
            if (raw < 0 || raw > long.MaxValue || raw != Math.Floor(raw))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON integer field is invalid: " + name);
            return (long)raw;
        }

        static Dictionary<string, object> ObjectValue(Dictionary<string, object> value, string name)
        {
            var result = Required(value, name) as Dictionary<string, object>;
            if (result == null)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON object field is invalid: " + name);
            return result;
        }

        static List<object> ArrayValue(Dictionary<string, object> value, string name)
        {
            var result = Required(value, name) as List<object>;
            if (result == null)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON array field is invalid: " + name);
            return result;
        }

        static string[] StringArray(Dictionary<string, object> value, string name)
        {
            List<object> raw = ArrayValue(value, name);
            var result = new string[raw.Count];
            for (int index = 0; index < raw.Count; index++)
            {
                result[index] = raw[index] as string;
                if (string.IsNullOrWhiteSpace(result[index]))
                    throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON string array is invalid: " + name);
            }
            return result;
        }

        /// <summary>Small strict RFC 8259 parser used to reject duplicate keys and non-standard numeric literals.</summary>
        internal static class StrictJson
        {
            public static Dictionary<string, object> ParseObject(byte[] bytes)
            {
                if (bytes == null) throw new ArgumentNullException(nameof(bytes));
                string text = new UTF8Encoding(false, true).GetString(bytes);
                var parser = new Parser(text);
                object value = parser.Parse();
                var result = value as Dictionary<string, object>;
                if (result == null) throw new InvalidDataException("JSON root must be an object");
                return result;
            }

            sealed class Parser
            {
                readonly string text;
                int index;
                int depth;

                public Parser(string text)
                {
                    this.text = text ?? throw new ArgumentNullException(nameof(text));
                }

                public object Parse()
                {
                    SkipWhitespace();
                    object value = ParseValue();
                    SkipWhitespace();
                    if (index != text.Length) throw Error("unexpected trailing JSON data");
                    return value;
                }

                object ParseValue()
                {
                    if (index >= text.Length) throw Error("unexpected end of JSON");
                    char c = text[index];
                    if (c == '{') return ParseObject();
                    if (c == '[') return ParseArray();
                    if (c == '"') return ParseString();
                    if (c == 't') return ParseLiteral("true", true);
                    if (c == 'f') return ParseLiteral("false", false);
                    if (c == 'n') return ParseLiteral("null", null);
                    if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();
                    throw Error("invalid JSON value");
                }

                Dictionary<string, object> ParseObject()
                {
                    Enter();
                    index++;
                    var result = new Dictionary<string, object>(StringComparer.Ordinal);
                    SkipWhitespace();
                    if (Take('}')) { Exit(); return result; }
                    while (true)
                    {
                        SkipWhitespace();
                        if (index >= text.Length || text[index] != '"') throw Error("JSON object key must be a string");
                        string key = ParseString();
                        SkipWhitespace();
                        Expect(':');
                        SkipWhitespace();
                        object value = ParseValue();
                        if (!result.TryAdd(key, value)) throw Error("duplicate JSON key: " + key);
                        SkipWhitespace();
                        if (Take('}')) { Exit(); return result; }
                        Expect(',');
                    }
                }

                List<object> ParseArray()
                {
                    Enter();
                    index++;
                    var result = new List<object>();
                    SkipWhitespace();
                    if (Take(']')) { Exit(); return result; }
                    while (true)
                    {
                        SkipWhitespace();
                        result.Add(ParseValue());
                        SkipWhitespace();
                        if (Take(']')) { Exit(); return result; }
                        Expect(',');
                    }
                }

                string ParseString()
                {
                    Expect('"');
                    var builder = new StringBuilder();
                    while (index < text.Length)
                    {
                        char c = text[index++];
                        if (c == '"') return builder.ToString();
                        if (c < 0x20) throw Error("unescaped control character in JSON string");
                        if (c != '\\') { builder.Append(c); continue; }
                        if (index >= text.Length) throw Error("truncated JSON escape");
                        char escaped = text[index++];
                        switch (escaped)
                        {
                            case '"': builder.Append('"'); break;
                            case '\\': builder.Append('\\'); break;
                            case '/': builder.Append('/'); break;
                            case 'b': builder.Append('\b'); break;
                            case 'f': builder.Append('\f'); break;
                            case 'n': builder.Append('\n'); break;
                            case 'r': builder.Append('\r'); break;
                            case 't': builder.Append('\t'); break;
                            case 'u':
                                if (index + 4 > text.Length) throw Error("truncated JSON Unicode escape");
                                string hex = text.Substring(index, 4);
                                ushort code;
                                if (!ushort.TryParse(hex, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out code))
                                    throw Error("invalid JSON Unicode escape");
                                builder.Append((char)code);
                                index += 4;
                                break;
                            default: throw Error("invalid JSON escape");
                        }
                    }
                    throw Error("unterminated JSON string");
                }

                double ParseNumber()
                {
                    int start = index;
                    if (Take('-') && index >= text.Length) throw Error("truncated JSON number");
                    if (Take('0'))
                    {
                        if (index < text.Length && char.IsDigit(text[index])) throw Error("leading zero in JSON number");
                    }
                    else
                    {
                        if (index >= text.Length || text[index] < '1' || text[index] > '9') throw Error("invalid JSON number");
                        while (index < text.Length && char.IsDigit(text[index])) index++;
                    }
                    if (Take('.'))
                    {
                        if (index >= text.Length || !char.IsDigit(text[index])) throw Error("invalid JSON fraction");
                        while (index < text.Length && char.IsDigit(text[index])) index++;
                    }
                    if (index < text.Length && (text[index] == 'e' || text[index] == 'E'))
                    {
                        index++;
                        if (index < text.Length && (text[index] == '+' || text[index] == '-')) index++;
                        if (index >= text.Length || !char.IsDigit(text[index])) throw Error("invalid JSON exponent");
                        while (index < text.Length && char.IsDigit(text[index])) index++;
                    }
                    double value;
                    if (!double.TryParse(text.Substring(start, index - start), NumberStyles.Float,
                        CultureInfo.InvariantCulture, out value) || double.IsNaN(value) || double.IsInfinity(value))
                        throw Error("non-finite JSON number");
                    return value;
                }

                object ParseLiteral(string literal, object value)
                {
                    if (index + literal.Length > text.Length ||
                        !string.Equals(text.Substring(index, literal.Length), literal, StringComparison.Ordinal))
                        throw Error("invalid JSON literal");
                    index += literal.Length;
                    return value;
                }

                void Enter()
                {
                    depth++;
                    if (depth > 64) throw Error("JSON nesting is too deep");
                }

                void Exit() { depth--; }
                bool Take(char expected)
                {
                    if (index < text.Length && text[index] == expected) { index++; return true; }
                    return false;
                }
                void Expect(char expected)
                {
                    if (!Take(expected)) throw Error("expected JSON token " + expected);
                }
                void SkipWhitespace()
                {
                    while (index < text.Length && (text[index] == ' ' || text[index] == '\t' || text[index] == '\r' || text[index] == '\n')) index++;
                }
                InvalidDataException Error(string message) { return new InvalidDataException(message + " at offset " + index); }
            }
        }
    }

    public sealed class PersonalRuntimeTemplateDistance
    {
        public string Word { get; }
        public string PrototypeId { get; }
        public double Distance { get; }
        public int PathLength { get; }

        internal PersonalRuntimeTemplateDistance(string word, string prototypeId, double distance, int pathLength)
        {
            Word = word;
            PrototypeId = prototypeId;
            Distance = distance;
            PathLength = pathLength;
        }
    }

    public sealed class PersonalRuntimeRecognitionResult
    {
        readonly IReadOnlyDictionary<string, double> scores;
        readonly IReadOnlyDictionary<string, bool> perClassAccept;

        public string QuerySessionId { get; }
        public string ExpectedWord { get; }
        public string PredictedWord { get; }
        public string WinningPrototypeId { get; }
        public double BestDistance { get; }
        public bool Accepted { get; }
        public bool ExpectedWordClassAccept { get; }
        public bool QuestSuccess { get; }
        public IReadOnlyDictionary<string, double> Scores => scores;
        public IReadOnlyDictionary<string, bool> PerClassAccept => perClassAccept;
        public IReadOnlyList<PersonalRuntimeTemplateDistance> TemplateDistances { get; }
        public IReadOnlyList<string> Warnings { get; }

        internal PersonalRuntimeRecognitionResult(
            string querySessionId,
            string expectedWord,
            string predictedWord,
            string winningPrototypeId,
            double bestDistance,
            bool accepted,
            bool expectedWordClassAccept,
            bool questSuccess,
            IDictionary<string, double> scores,
            IDictionary<string, bool> perClassAccept,
            IEnumerable<PersonalRuntimeTemplateDistance> templateDistances,
            IReadOnlyList<string> warnings)
        {
            QuerySessionId = querySessionId;
            ExpectedWord = expectedWord;
            PredictedWord = predictedWord;
            WinningPrototypeId = winningPrototypeId;
            BestDistance = bestDistance;
            Accepted = accepted;
            ExpectedWordClassAccept = expectedWordClassAccept;
            QuestSuccess = questSuccess;
            this.scores = new ReadOnlyDictionary<string, double>(new Dictionary<string, double>(scores, StringComparer.Ordinal));
            this.perClassAccept = new ReadOnlyDictionary<string, bool>(new Dictionary<string, bool>(perClassAccept, StringComparer.Ordinal));
            TemplateDistances = Array.AsReadOnly(templateDistances.ToArray());
            Warnings = warnings;
        }
    }

    public interface IPersonalRuntimeEvaluator
    {
        PersonalRuntimeRecognitionResult Recognize(string querySessionId, RecognitionSequence query, string expectedWord);
    }

    /// <summary>Three-medoid evaluator that reuses the approved MaskAwareDtw implementation.</summary>
    public sealed class PersonalRuntimeEvaluator : IPersonalRuntimeEvaluator
    {
        readonly PersonalRuntimeProfile profile;

        public PersonalRuntimeEvaluator(PersonalRuntimeProfile profile)
        {
            this.profile = profile ?? throw new ArgumentNullException(nameof(profile));
        }

        public PersonalRuntimeRecognitionResult Recognize(string querySessionId, RecognitionSequence query, string expectedWord)
        {
            if (string.IsNullOrWhiteSpace(querySessionId)) throw new ArgumentException("query/session ID is required", nameof(querySessionId));
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (query.SampleId != querySessionId) throw new ArgumentException("query/session ID mismatch", nameof(querySessionId));
            if (query.Split == "train") throw new ArgumentException("a global train template cannot be used as a personal runtime query", nameof(query));
            if (!string.IsNullOrEmpty(expectedWord) && !profile.Words.Contains(expectedWord, StringComparer.Ordinal))
                throw new ArgumentException("unsupported expected word", nameof(expectedWord));

            var scores = new Dictionary<string, double>(StringComparer.Ordinal);
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal);
            var distances = new List<PersonalRuntimeTemplateDistance>();
            foreach (string word in profile.Words)
            {
                RecognitionSequence prototype = profile.Templates[word];
                DtwResult dtw = MaskAwareDtw.Compute(query, prototype, RecognitionCore.RelativeBand);
                scores.Add(word, dtw.Distance);
                accepts.Add(word, dtw.Distance <= profile.Thresholds[word]);
                distances.Add(new PersonalRuntimeTemplateDistance(word, prototype.SampleId, dtw.Distance, dtw.PathLength));
            }

            string predicted = profile.Words[0];
            foreach (string word in profile.Words.Skip(1))
                if (scores[word] < scores[predicted] ||
                    (scores[word] == scores[predicted] && string.CompareOrdinal(word, predicted) < 0))
                    predicted = word;
            bool accepted = accepts[predicted];
            bool expectedAccept = !string.IsNullOrEmpty(expectedWord) && accepts[expectedWord];
            bool questSuccess = accepted && predicted == expectedWord && expectedAccept;
            return new PersonalRuntimeRecognitionResult(
                querySessionId, expectedWord, predicted, profile.Templates[predicted].SampleId,
                scores[predicted], accepted, expectedAccept, questSuccess,
                scores, accepts, distances, profile.Warnings);
        }
    }

    public enum RuntimeOptInMode
    {
        Global,
        Personal,
    }

    /// <summary>Discriminated immutable result preserving the existing global result object byte-for-byte in global mode.</summary>
    public sealed class RuntimeOptInRecognitionResult
    {
        public RuntimeOptInMode Mode { get; }
        public RuntimeRecognitionResult GlobalResult { get; }
        public PersonalRuntimeRecognitionResult PersonalResult { get; }
        public string PredictedWord { get; }
        public bool Accepted { get; }
        public bool QuestSuccess { get; }

        internal RuntimeOptInRecognitionResult(RuntimeRecognitionResult global, string expectedWord)
        {
            Mode = RuntimeOptInMode.Global;
            GlobalResult = global ?? throw new ArgumentNullException(nameof(global));
            PersonalResult = null;
            PredictedWord = global.PredictedWord;
            Accepted = global.Accepted;
            QuestSuccess = global.Accepted && !string.IsNullOrEmpty(expectedWord) && global.PredictedWord == expectedWord;
        }

        internal RuntimeOptInRecognitionResult(PersonalRuntimeRecognitionResult personal)
        {
            Mode = RuntimeOptInMode.Personal;
            GlobalResult = null;
            PersonalResult = personal ?? throw new ArgumentNullException(nameof(personal));
            PredictedWord = personal.PredictedWord;
            Accepted = personal.Accepted;
            QuestSuccess = personal.QuestSuccess;
        }
    }

    /// <summary>
    /// Explicit session-level switch. Construction is global-only; activation is atomic and fail-closed;
    /// deactivation returns to the unchanged RuntimeRecognitionService.
    /// </summary>
    public sealed class PersonalRuntimeOptInSession
    {
        readonly RuntimeRecognitionService globalService;
        PersonalRuntimeProfile profile;
        IPersonalRuntimeEvaluator personalEvaluator;
        string activeProfileId = string.Empty;

        public RuntimeOptInMode Mode => personalEvaluator == null ? RuntimeOptInMode.Global : RuntimeOptInMode.Personal;
        public bool IsPersonalActive => personalEvaluator != null;
        public string ActiveProfileId => activeProfileId;

        public PersonalRuntimeOptInSession(RuntimeRecognitionService globalService)
        {
            this.globalService = globalService ?? throw new ArgumentNullException(nameof(globalService));
        }

        public void ActivatePersonalProfile(string profileDirectory)
        {
            PersonalRuntimeProfile candidate = PersonalRuntimeProfileLoader.Load(profileDirectory);
            var candidateEvaluator = new PersonalRuntimeEvaluator(candidate);
            profile = candidate;
            personalEvaluator = candidateEvaluator;
            activeProfileId = candidate.ProfileId;
        }

        public void ActivatePersonalMultiPrototypeProfile(string profileDirectory)
        {
            PersonalMultiPrototypeRuntimeProfile candidate = PersonalMultiPrototypeRuntimeProfileLoader.Load(profileDirectory);
            var candidateEvaluator = new PersonalMultiPrototypeRuntimeEvaluator(candidate);
            profile = null;
            personalEvaluator = candidateEvaluator;
            activeProfileId = candidate.ProfileId;
        }

        public void DeactivatePersonalProfile()
        {
            personalEvaluator = null;
            profile = null;
            activeProfileId = string.Empty;
        }

        public RuntimeOptInRecognitionResult Recognize(string querySessionId, RecognitionSequence query, string expectedWord = "")
        {
            if (personalEvaluator == null)
                return new RuntimeOptInRecognitionResult(globalService.Recognize(querySessionId, query), expectedWord);
            return new RuntimeOptInRecognitionResult(personalEvaluator.Recognize(querySessionId, query, expectedWord));
        }
    }
}
