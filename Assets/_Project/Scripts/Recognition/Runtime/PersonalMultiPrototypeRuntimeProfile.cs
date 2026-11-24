using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Recognition.Runtime
{
    public sealed class PersonalMultiPrototypeRuntimeProfile
    {
        readonly IReadOnlyDictionary<string, IReadOnlyList<RecognitionSequence>> prototypes;
        readonly IReadOnlyDictionary<string, IReadOnlyList<PersonalTemplateProvenance>> provenance;
        readonly IReadOnlyDictionary<string, double> thresholds;

        public string ProfileId { get; }
        public bool OptInOnly => true;
        public bool DefaultActive => false;
        public IReadOnlyList<string> Words { get; }
        public IReadOnlyDictionary<string, IReadOnlyList<RecognitionSequence>> Prototypes => prototypes;
        public IReadOnlyDictionary<string, IReadOnlyList<PersonalTemplateProvenance>> PrototypeProvenance => provenance;
        public IReadOnlyDictionary<string, double> Thresholds => thresholds;
        public IReadOnlyList<string> Warnings { get; }

        internal PersonalMultiPrototypeRuntimeProfile(
            string profileId,
            IDictionary<string, List<RecognitionSequence>> prototypes,
            IDictionary<string, List<PersonalTemplateProvenance>> provenance,
            IDictionary<string, double> thresholds,
            IEnumerable<string> warnings)
        {
            ProfileId = profileId;
            this.prototypes = new ReadOnlyDictionary<string, IReadOnlyList<RecognitionSequence>>(
                prototypes.ToDictionary(
                    pair => pair.Key,
                    pair => (IReadOnlyList<RecognitionSequence>)Array.AsReadOnly(pair.Value.ToArray()),
                    StringComparer.Ordinal));
            this.provenance = new ReadOnlyDictionary<string, IReadOnlyList<PersonalTemplateProvenance>>(
                provenance.ToDictionary(
                    pair => pair.Key,
                    pair => (IReadOnlyList<PersonalTemplateProvenance>)Array.AsReadOnly(pair.Value.ToArray()),
                    StringComparer.Ordinal));
            this.thresholds = new ReadOnlyDictionary<string, double>(
                new Dictionary<string, double>(thresholds, StringComparer.Ordinal));
            Words = Array.AsReadOnly(PersonalRuntimeProfileLoader.RequiredWords.ToArray());
            Warnings = Array.AsReadOnly(warnings.ToArray());
        }
    }

    public static class PersonalMultiPrototypeRuntimeProfileLoader
    {
        public const string ProfileId = "step9l_current_user_multi_prototype_v1_0";
        public const string ManifestFileName = "personal_runtime_profile_manifest.json";
        public const string ConfigFileName = "personal_multi_prototype_threshold_config.json";
        public const string CandidateWarning = "STEP9L_MULTI_PROTOTYPE_CANDIDATE_NOT_LIVE_VALIDATED";
        public const string Step9HDeliverySha256 = "4827e81e029e63e7939e7ed91cb33ac5f90dd6822fa13722da01f642aed2ddb9";
        public const long Step9HDeliverySizeBytes = 3214867;
        public const string ConfigFileSha256 = "42720034661a1fb9f6c673918f55fb0da3bf0507555463d4a13170718e81bfe0";
        public const long ConfigFileSizeBytes = 12170;

        sealed class ExpectedPrototype
        {
            public string Word;
            public string FileName;
            public long SizeBytes;
            public string FileSha256;
            public string EmbeddedChecksumSha256;
            public string CaptureId;
            public string CaptureKey;
            public string RunId;
            public string SourceFamily;
            public long FrameCount;
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
            public string SourceFamily;
            public long FrameCount;
            public string SourcePayloadSha256;
            public string CoordinatePayloadSha256;
            public string MaskPayloadSha256;
            public string NormalizedPayloadSha256;
        }

        static readonly ExpectedPrototype[] ExpectedPrototypes =
        {
            new ExpectedPrototype
            {
                Word = "GREETING",
                FileName = "personal_prototype_GREETING_step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7.wisrt",
                SizeBytes = 54750,
                FileSha256 = "3ac4ec3e6468787c4f203503c9481c5f45bf245b1c4d4dc3d1e0841e6ff848fa",
                EmbeddedChecksumSha256 = "9f63dc0874c26e20429a08f58fef0413bbfc8e6d04d3d0a2da8094156cf45055",
                CaptureId = "step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7",
                CaptureKey = "baseline_step9c::step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7",
                RunId = "8d17ef49-89ff-4d40-ae59-0386cc406e2e",
                SourceFamily = "baseline_step9c_physical_query",
                FrameCount = 141,
                SourcePayloadSha256 = "3ed50fc5be7f5c3e30417aa6656b10470f0b2d67c09a2f79644af83cc14d5a69",
                CoordinatePayloadSha256 = "244e01dc1ad192d06822547da3d30e00e23371a9bccb4a8580373fbaf33b2acf",
                MaskPayloadSha256 = "dd1daa7dc1698ff32cd151577d5f23b9227de634d16cae5383be6634ff3c3600",
                NormalizedPayloadSha256 = "68d2ead4124a7023a3ff628b5e51a63e61c304932b1465b243a8c10256003e38",
            },
            new ExpectedPrototype
            {
                Word = "GREETING",
                FileName = "personal_prototype_GREETING_26f436475e2f4e8c832dbce00cbfe883.wisrt",
                SizeBytes = 48546,
                FileSha256 = "d5b56a6c714f22646b47e819a7a5d124433ca62b03f5968bfc8c9cf48c8d4a84",
                EmbeddedChecksumSha256 = "7fc87d2de875ebe8763d2eb72d6f3ce93ee282ac2663532b51aa5d396ea2365b",
                CaptureId = "26f436475e2f4e8c832dbce00cbfe883",
                CaptureKey = "raw_intermediate_step9d::26f436475e2f4e8c832dbce00cbfe883",
                RunId = "e965f94530f6432c800eea60be8f2cdf",
                SourceFamily = "step9d_raw_intermediate_capture",
                FrameCount = 125,
                SourcePayloadSha256 = "bc74d8cb87fae7600bbabca0aabb97bad17c4128949f15513d871ed1380f9263",
                CoordinatePayloadSha256 = "23125988f7ddeb2619d57607a0732b02d539504328c17d9807c83e332cd36532",
                MaskPayloadSha256 = "8262d61830c874f65537739711611a891194806b5b1a4ae201d0e83c320974fe",
                NormalizedPayloadSha256 = "43b69177f2deace99f55c58b3e8c07619c68b47f414cb94383e2b3a77cabeb15",
            },
            new ExpectedPrototype
            {
                Word = "GREETING",
                FileName = "personal_prototype_GREETING_87e5b54432ac49a197a596d5ff7c3845.wisrt",
                SizeBytes = 53564,
                FileSha256 = "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681",
                EmbeddedChecksumSha256 = "bd9c9c421f51df4e89ae9119427ac555e513a025ecdc6ec48e1d7858e96e3a26",
                CaptureId = "87e5b54432ac49a197a596d5ff7c3845",
                CaptureKey = "step9f_additional::87e5b54432ac49a197a596d5ff7c3845",
                RunId = "3698e6b5a88d438d9f94b1742978eb36",
                SourceFamily = "step9f_additional_raw_intermediate_capture",
                FrameCount = 138,
                SourcePayloadSha256 = "146897a4769bb164b9fdb013a6897437462a423c1fe9ef15cb6ce9d4cdbcc240",
                CoordinatePayloadSha256 = "0b6c1ac2cfc4422af014a79f49acfdf9bfe53bf36a1c08c80c336fd978d3618e",
                MaskPayloadSha256 = "d8e56cd267758595645f298a3dd3ad665943ced8a583918754a9add21c4f731d",
                NormalizedPayloadSha256 = "911dc3db1f079ed0010dbfe3a431c32f6bbbd7fc77c1803b05f8a1bc03c9d975",
            },
            new ExpectedPrototype
            {
                Word = "LOVE",
                FileName = "personal_prototype_LOVE_step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2.wisrt",
                SizeBytes = 62462,
                FileSha256 = "6117c8204a2193b60639a0629a4006f6200c79862bca5fe372e08457b345a731",
                EmbeddedChecksumSha256 = "7cbcd2b1384249a31a499359c4c896dd96d00f4e45e6cd494223fc53d3f4a08c",
                CaptureId = "step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2",
                CaptureKey = "baseline_step9c::step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2",
                RunId = "8d17ef49-89ff-4d40-ae59-0386cc406e2e",
                SourceFamily = "baseline_step9c_physical_query",
                FrameCount = 161,
                SourcePayloadSha256 = "b85b202cad78f51230ac8ded1b98632dbe5012e5772b957328d1cb85f95e21a2",
                CoordinatePayloadSha256 = "5517ddaa39c74612f16984e97e222302aaa2ae049655d5630ea54784e27e2420",
                MaskPayloadSha256 = "9d9edc8ef9cebbb6498d19e8de9347e1487d84a67312d63d596497759d588f08",
                NormalizedPayloadSha256 = "6472b9dbe35827ee4d0409cc78e2569531c16feb633cd90652951fe8afd0d583",
            },
            new ExpectedPrototype
            {
                Word = "LOVE",
                FileName = "personal_prototype_LOVE_cd65ad6b277c48b78fa50785070fdddc.wisrt",
                SizeBytes = 63210,
                FileSha256 = "5c17605bbc922f27c1ac54a97017e72c5812e89ae692aa4b81cf2139f7ccc2d8",
                EmbeddedChecksumSha256 = "cb65a47f2ab124987bbee7f19b34f875ba91868ce2464fdda0687b6f7b54a30b",
                CaptureId = "cd65ad6b277c48b78fa50785070fdddc",
                CaptureKey = "raw_intermediate_step9d::cd65ad6b277c48b78fa50785070fdddc",
                RunId = "e965f94530f6432c800eea60be8f2cdf",
                SourceFamily = "step9d_raw_intermediate_capture",
                FrameCount = 163,
                SourcePayloadSha256 = "b81e8cd8edf2a5c592369848fa5f68e2bedb7e7bb05370db55513b5b7e2e1179",
                CoordinatePayloadSha256 = "c28ad5fa7036a2ce43ad162870e2956d9193f5d9a279d7e64b7bd99a49f3732d",
                MaskPayloadSha256 = "9e6f73a48b61071b6bde0656008d2622f9d95f26b6f90d82d70cf2fa68c41a55",
                NormalizedPayloadSha256 = "78b02d6820d6d99459f31d84febad7cf505cc92e51b9bfdc89454c7277ff5e5c",
            },
            new ExpectedPrototype
            {
                Word = "LOVE",
                FileName = "personal_prototype_LOVE_d958cbc2a1c44e03bb29bf64a6fae195.wisrt",
                SizeBytes = 56262,
                FileSha256 = "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b",
                EmbeddedChecksumSha256 = "d88398c029222013da4be539172a47f8dbb9ca2e39645b937518bc6bfd6d98e8",
                CaptureId = "d958cbc2a1c44e03bb29bf64a6fae195",
                CaptureKey = "step9f_additional::d958cbc2a1c44e03bb29bf64a6fae195",
                RunId = "3698e6b5a88d438d9f94b1742978eb36",
                SourceFamily = "step9f_additional_raw_intermediate_capture",
                FrameCount = 145,
                SourcePayloadSha256 = "cd9f516cf4ae1575a6334577d5328dc843b93d330bca79f81d55ef03ba8393ad",
                CoordinatePayloadSha256 = "c6731402913b61b66eaa47f069e3f8e9f525e01e13756f6834c56be2db7b6ea4",
                MaskPayloadSha256 = "6e613d2fddf0efa3e136be41214504180db94c85c231aa071964f97836c0adc4",
                NormalizedPayloadSha256 = "5478fadab109b02faa260b7c2f3e58560968f81d773b98a00883a922f02648dc",
            },
            new ExpectedPrototype
            {
                Word = "THANK_YOU",
                FileName = "personal_prototype_THANK_YOU_step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42.wisrt",
                SizeBytes = 62858,
                FileSha256 = "cd2fc174c50c7f6c3e4adc089ac6ad12b2be60d90b8c003dc277c0b8597fdbe0",
                EmbeddedChecksumSha256 = "28f500333f7f92420f0391d62c676c1f1aa3a80787a187aecb9390a9d90433d5",
                CaptureId = "step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42",
                CaptureKey = "baseline_step9c::step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42",
                RunId = "8d17ef49-89ff-4d40-ae59-0386cc406e2e",
                SourceFamily = "baseline_step9c_physical_query",
                FrameCount = 162,
                SourcePayloadSha256 = "c30e68736fd9762864c6d5da3cb5e2c995060c0136a99e48391e8f5b5572f5c2",
                CoordinatePayloadSha256 = "4387c85d3b4efb4c4c1384c443e78a7e73da39b23cf5b632b99f6352d15d2105",
                MaskPayloadSha256 = "51f544a8259777ee4ebac3d41694cf6c33d66284e270fe1fde1596b07a18ad90",
                NormalizedPayloadSha256 = "ca39f7012dedb4d6d22da5526d78a50ae8e89d4d4fb8a7d91f44546ccad0ded4",
            },
            new ExpectedPrototype
            {
                Word = "THANK_YOU",
                FileName = "personal_prototype_THANK_YOU_2b323671e3f64049a74556c71ff70154.wisrt",
                SizeBytes = 49319,
                FileSha256 = "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5",
                EmbeddedChecksumSha256 = "1b05c1c912916bd0ed90ab43f50717c177a6b83a0a482cdfeafbb262a7f46e00",
                CaptureId = "2b323671e3f64049a74556c71ff70154",
                CaptureKey = "raw_intermediate_step9d::2b323671e3f64049a74556c71ff70154",
                RunId = "e965f94530f6432c800eea60be8f2cdf",
                SourceFamily = "step9d_raw_intermediate_capture",
                FrameCount = 127,
                SourcePayloadSha256 = "858042fa4b3474b366c0236e0a8693f366204530366656f18fff74a55289dfc9",
                CoordinatePayloadSha256 = "490dbeb1e50953dd0d51ad2de570ae2534f786af833fcadf0857d2e005a410d9",
                MaskPayloadSha256 = "919b9da5efa1ea9fd1bd17a8bcd8412d1c27bd2c790ef091e8ecd240ae85a113",
                NormalizedPayloadSha256 = "6223c805af9245254ee223ba832eaa916d85e9d2d923ff81e161f34503c63644",
            },
            new ExpectedPrototype
            {
                Word = "THANK_YOU",
                FileName = "personal_prototype_THANK_YOU_9926cef4a46649bea3139fbace2a0ff3.wisrt",
                SizeBytes = 44301,
                FileSha256 = "d6919f37afe725da6068654e446c1424ef132e451610c537c82f10bb4efb7fed",
                EmbeddedChecksumSha256 = "93b80536df5e442eb2b35c56c5aef27eef8bb8bad13a1a0dcc8c7ed5696718f6",
                CaptureId = "9926cef4a46649bea3139fbace2a0ff3",
                CaptureKey = "step9f_additional::9926cef4a46649bea3139fbace2a0ff3",
                RunId = "3698e6b5a88d438d9f94b1742978eb36",
                SourceFamily = "step9f_additional_raw_intermediate_capture",
                FrameCount = 114,
                SourcePayloadSha256 = "45fdbdd29e7919f8fb7918b10b7907fed9f54749c670ab78f7474210a5f7c067",
                CoordinatePayloadSha256 = "cf2088ee705713ddde5e4b8dfc05b1c4e67e356f82af67efd1819853b87039d1",
                MaskPayloadSha256 = "18a48878ee1ceddc418dd06b8e46257c4a73a485ab9eb660dacf0d8ce13a0587",
                NormalizedPayloadSha256 = "b2411682196c464ef07118290dbbdc815b7809222c6962cb32d7915780423fbe",
            },
        };

        static readonly IReadOnlyDictionary<string, double> ExpectedThresholds =
            new ReadOnlyDictionary<string, double>(new Dictionary<string, double>(StringComparer.Ordinal)
            {
                { "GREETING", 0.5113059473720765 },
                { "LOVE", 0.391276023460427 },
                { "THANK_YOU", 0.37709977431991587 },
            });

        public static PersonalMultiPrototypeRuntimeProfile Load(string profileDirectory)
        {
            if (string.IsNullOrWhiteSpace(profileDirectory))
                throw Failure(PersonalProfileFailureCode.ProfileDirectoryMissing, "multi-prototype profile directory is required");
            if (!Directory.Exists(RuntimeLongPath.ForIo(profileDirectory)))
                throw Failure(PersonalProfileFailureCode.ProfileDirectoryMissing, "multi-prototype profile directory is missing");

            string manifestPath = Path.Combine(profileDirectory, ManifestFileName);
            if (!File.Exists(RuntimeLongPath.ForIo(manifestPath)))
                throw Failure(PersonalProfileFailureCode.ManifestMissing, "multi-prototype profile manifest is missing");

            Dictionary<string, object> manifest;
            try { manifest = PersonalRuntimeProfileLoader.StrictJson.ParseObject(File.ReadAllBytes(RuntimeLongPath.ForIo(manifestPath))); }
            catch (Exception exc) { throw Failure(PersonalProfileFailureCode.ManifestMalformed, "multi-prototype manifest is malformed", exc); }

            try
            {
                ValidateManifestHeader(manifest);
                List<ManifestAsset> assets = ReadManifestAssets(manifest);
                ManifestAsset configAsset = assets.Single(item => item.Kind == "threshold_config");
                Dictionary<string, ManifestAsset> prototypeAssets = assets
                    .Where(item => item.Kind == "prototype")
                    .ToDictionary(item => item.CaptureId, StringComparer.Ordinal);

                byte[] configBytes = ReadAsset(ResolveSafe(profileDirectory, configAsset.RelativePath), configAsset);
                Dictionary<string, double> thresholds;
                Dictionary<string, Dictionary<string, object>> configRecords;
                ValidateConfig(configBytes, out thresholds, out configRecords);
                ValidateAssetSha(configBytes, configAsset, ConfigFileSha256);

                var prototypes = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                    word => word, word => new List<RecognitionSequence>(), StringComparer.Ordinal);
                var provenance = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                    word => word, word => new List<PersonalTemplateProvenance>(), StringComparer.Ordinal);
                foreach (ExpectedPrototype expected in ExpectedPrototypes)
                {
                    ManifestAsset asset;
                    if (!prototypeAssets.TryGetValue(expected.CaptureId, out asset))
                        throw Failure(PersonalProfileFailureCode.WordSetMismatch, "multi-prototype capture is missing: " + expected.CaptureId);
                    ValidateManifestPrototype(asset, expected);
                    byte[] bytes = ReadAsset(ResolveSafe(profileDirectory, asset.RelativePath), asset);
                    RecognitionSequence sequence;
                    try { sequence = RecognitionSequenceBinary.Load(bytes); }
                    catch (InvalidDataException exc)
                    {
                        PersonalProfileFailureCode code = exc.Message.IndexOf("payload SHA-256", StringComparison.Ordinal) >= 0
                            ? PersonalProfileFailureCode.EmbeddedChecksumMismatch
                            : PersonalProfileFailureCode.WisrtFormatInvalid;
                        throw Failure(code, "multi-prototype WISRT validation failed: " + expected.CaptureId, exc);
                    }
                    ValidateAssetSha(bytes, asset, expected.FileSha256);
                    ValidateSequence(sequence, expected, configRecords[expected.CaptureId]);
                    prototypes[expected.Word].Add(sequence);
                    provenance[expected.Word].Add(new PersonalTemplateProvenance(
                        expected.Word, expected.FileName, expected.SizeBytes, expected.FileSha256,
                        expected.EmbeddedChecksumSha256, expected.CaptureId, expected.CaptureKey, expected.RunId,
                        expected.SourcePayloadSha256, expected.CoordinatePayloadSha256,
                        expected.MaskPayloadSha256, expected.NormalizedPayloadSha256));
                }
                if (prototypes.Any(pair => pair.Value.Count != 3) || prototypes.Sum(pair => pair.Value.Count) != 9)
                    throw Failure(PersonalProfileFailureCode.WordSetMismatch, "multi-prototype profile must contain three prototypes per word");
                if (Directory.GetFiles(profileDirectory, "*.wisrt", SearchOption.TopDirectoryOnly).Length != 9)
                    throw Failure(PersonalProfileFailureCode.WordSetMismatch, "multi-prototype profile must contain exactly nine WISRT files");
                return new PersonalMultiPrototypeRuntimeProfile(
                    ProfileId, prototypes, provenance, thresholds, new[] { CandidateWarning });
            }
            catch (PersonalProfileLoadException) { throw; }
            catch (Exception exc)
            {
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "multi-prototype profile validation failed", exc);
            }
        }

        static void ValidateManifestHeader(Dictionary<string, object> manifest)
        {
            ExactKeys(manifest, "schema_version", "profile_id", "mode", "score_rule", "threshold_mode",
                "opt_in_only", "default_active", "installed_but_default_inactive", "prototype_count",
                "prototype_count_per_word", "threshold_count", "words", "source_step9h_delivery", "assets",
                "global_assets_overwritten", "existing_step9i_profile_overwritten", "warnings");
            RequireEqual(StringValue(manifest, "schema_version"), "words_in_silence.step9l.personal_multi_prototype_runtime_candidate.v1_0.personal_runtime_profile_manifest", "multi-prototype manifest schema mismatch");
            RequireEqual(StringValue(manifest, "profile_id"), ProfileId, "multi-prototype profile ID mismatch");
            RequireEqual(StringValue(manifest, "mode"), "personal_multi_prototype_candidate", "multi-prototype mode mismatch");
            RequireEqual(StringValue(manifest, "score_rule"), "minimum_frozen_mask_aware_dtw_distance_per_class", "multi-prototype score rule mismatch");
            RequireEqual(StringValue(manifest, "threshold_mode"), "strict_leave_one_out_class_conditional_midpoint", "multi-prototype threshold mode mismatch");
            if (!BoolValue(manifest, "opt_in_only") || BoolValue(manifest, "default_active") ||
                !BoolValue(manifest, "installed_but_default_inactive") || BoolValue(manifest, "global_assets_overwritten") ||
                BoolValue(manifest, "existing_step9i_profile_overwritten"))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "multi-prototype activation/preservation flags are invalid");
            if (Int64Value(manifest, "prototype_count") != 9 || Int64Value(manifest, "threshold_count") != 3)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "multi-prototype manifest counts are invalid");
            ValidateCountMap(ObjectValue(manifest, "prototype_count_per_word"));
            ValidateWords(StringArray(manifest, "words"));
            string[] warnings = StringArray(manifest, "warnings");
            if (warnings.Length != 1 || warnings[0] != CandidateWarning)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "multi-prototype warning set is invalid");
            Dictionary<string, object> source = ObjectValue(manifest, "source_step9h_delivery");
            ExactKeys(source, "relative_path", "size_bytes", "sha256");
            RequireEqual(StringValue(source, "relative_path"), "Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip", "Step 9H source path mismatch");
            RequireEqual(StringValue(source, "sha256"), Step9HDeliverySha256, "Step 9H source SHA-256 mismatch");
            if (Int64Value(source, "size_bytes") != Step9HDeliverySizeBytes)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "Step 9H source size mismatch");
        }

        static List<ManifestAsset> ReadManifestAssets(Dictionary<string, object> manifest)
        {
            List<object> raw = ArrayValue(manifest, "assets");
            if (raw.Count != 10)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "ten multi-prototype assets are required");
            var result = new List<ManifestAsset>();
            var unique = new HashSet<string>(StringComparer.Ordinal);
            foreach (object value in raw)
            {
                var record = value as Dictionary<string, object>;
                if (record == null) throw Failure(PersonalProfileFailureCode.ManifestMalformed, "multi-prototype asset is not an object");
                string kind = StringValue(record, "kind");
                if (kind == "prototype")
                {
                    ExactKeys(record, "kind", "word", "capture_id", "capture_key", "run_id", "source_family",
                        "relative_path", "size_bytes", "sha256", "embedded_checksum_sha256", "source_payload_sha256",
                        "coordinate_payload_sha256", "mask_payload_sha256", "normalized_payload_sha256", "frame_count");
                    var asset = new ManifestAsset
                    {
                        Kind = kind, Word = StringValue(record, "word"), CaptureId = StringValue(record, "capture_id"),
                        CaptureKey = StringValue(record, "capture_key"), RunId = StringValue(record, "run_id"),
                        SourceFamily = StringValue(record, "source_family"), FrameCount = Int64Value(record, "frame_count"),
                        RelativePath = StringValue(record, "relative_path"), SizeBytes = Int64Value(record, "size_bytes"),
                        Sha256 = StringValue(record, "sha256"), EmbeddedChecksumSha256 = StringValue(record, "embedded_checksum_sha256"),
                        SourcePayloadSha256 = StringValue(record, "source_payload_sha256"),
                        CoordinatePayloadSha256 = StringValue(record, "coordinate_payload_sha256"),
                        MaskPayloadSha256 = StringValue(record, "mask_payload_sha256"),
                        NormalizedPayloadSha256 = StringValue(record, "normalized_payload_sha256"),
                    };
                    if (!unique.Add("prototype|" + asset.CaptureId))
                        throw Failure(PersonalProfileFailureCode.DuplicateWord, "duplicate multi-prototype capture ID");
                    result.Add(asset);
                }
                else if (kind == "threshold_config")
                {
                    ExactKeys(record, "kind", "relative_path", "size_bytes", "sha256");
                    var asset = new ManifestAsset
                    {
                        Kind = kind, RelativePath = StringValue(record, "relative_path"),
                        SizeBytes = Int64Value(record, "size_bytes"), Sha256 = StringValue(record, "sha256"),
                    };
                    if (!unique.Add("threshold_config"))
                        throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "duplicate threshold config asset");
                    result.Add(asset);
                }
                else throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "unsupported multi-prototype asset kind");
            }
            if (result.Count(item => item.Kind == "prototype") != 9 || result.Count(item => item.Kind == "threshold_config") != 1)
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "multi-prototype asset membership mismatch");
            return result;
        }

        static void ValidateConfig(byte[] bytes, out Dictionary<string, double> thresholds,
            out Dictionary<string, Dictionary<string, object>> records)
        {
            Dictionary<string, object> config;
            try { config = PersonalRuntimeProfileLoader.StrictJson.ParseObject(bytes); }
            catch (Exception exc) { throw Failure(PersonalProfileFailureCode.ConfigMalformed, "multi-prototype config is malformed", exc); }
            RequireEqual(StringValue(config, "schema_version"), "words_in_silence.step9l.personal_multi_prototype_runtime_candidate.v1_0.personal_multi_prototype_threshold_config", "multi-prototype config schema mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            RequireEqual(StringValue(config, "profile_id"), ProfileId, "multi-prototype config profile mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            RequireEqual(StringValue(config, "mode"), "personal_multi_prototype_candidate", "multi-prototype config mode mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            RequireEqual(StringValue(config, "score_rule"), "minimum_frozen_mask_aware_dtw_distance_per_class", "multi-prototype score rule mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            RequireEqual(StringValue(config, "threshold_mode"), "strict_leave_one_out_class_conditional_midpoint", "multi-prototype config threshold mode mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            if (!BoolValue(config, "opt_in_only") || BoolValue(config, "default_active") || BoolValue(config, "runtime_activated") ||
                BoolValue(config, "global_template_or_threshold_overwritten") || !BoolValue(config, "strict_loo_all_self_excluded"))
                throw Failure(PersonalProfileFailureCode.ConfigSchemaMismatch, "multi-prototype config flags are invalid");
            if (Int64Value(config, "source_capture_count") != 9 || Int64Value(config, "prototype_count") != 9 ||
                Int64Value(config, "threshold_count") != 3 || Int64Value(config, "strict_loo_prediction_correct_count") != 9)
                throw Failure(PersonalProfileFailureCode.ConfigSchemaMismatch, "multi-prototype config counts are invalid");
            RequireEqual(StringValue(config, "strict_loo_self_exclusion_identity"), "capture_id", "strict LOO exclusion identity mismatch", PersonalProfileFailureCode.ConfigSchemaMismatch);
            ValidateCountMap(ObjectValue(config, "prototype_count_per_word"));
            Dictionary<string, object> thresholdObject = ObjectValue(config, "thresholds");
            ExactKeys(thresholdObject, PersonalRuntimeProfileLoader.RequiredWords);
            thresholds = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (string word in PersonalRuntimeProfileLoader.RequiredWords)
            {
                double value = NumberValue(thresholdObject, word);
                if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0 || value != ExpectedThresholds[word])
                    throw Failure(PersonalProfileFailureCode.ThresholdInvalid, "multi-prototype threshold mismatch: " + word);
                thresholds.Add(word, value);
            }
            List<object> rawRecords = ArrayValue(config, "prototype_records");
            if (rawRecords.Count != 9) throw Failure(PersonalProfileFailureCode.WordSetMismatch, "multi-prototype config must contain nine records");
            records = new Dictionary<string, Dictionary<string, object>>(StringComparer.Ordinal);
            foreach (object raw in rawRecords)
            {
                var record = raw as Dictionary<string, object>;
                if (record == null) throw Failure(PersonalProfileFailureCode.ConfigMalformed, "multi-prototype config record is not an object");
                string captureId = StringValue(record, "capture_id");
                if (!records.TryAdd(captureId, record)) throw Failure(PersonalProfileFailureCode.DuplicateWord, "duplicate config capture ID");
            }
            string[] warnings = StringArray(config, "warnings");
            if (warnings.Length != 1 || warnings[0] != CandidateWarning)
                throw Failure(PersonalProfileFailureCode.ConfigSchemaMismatch, "multi-prototype config warning mismatch");
        }

        static void ValidateManifestPrototype(ManifestAsset asset, ExpectedPrototype expected)
        {
            if (asset.Word != expected.Word || asset.RelativePath != expected.FileName || asset.SizeBytes != expected.SizeBytes ||
                asset.Sha256 != expected.FileSha256 || asset.EmbeddedChecksumSha256 != expected.EmbeddedChecksumSha256 ||
                asset.CaptureId != expected.CaptureId || asset.CaptureKey != expected.CaptureKey || asset.RunId != expected.RunId ||
                asset.SourceFamily != expected.SourceFamily || asset.FrameCount != expected.FrameCount ||
                asset.SourcePayloadSha256 != expected.SourcePayloadSha256 || asset.CoordinatePayloadSha256 != expected.CoordinatePayloadSha256 ||
                asset.MaskPayloadSha256 != expected.MaskPayloadSha256 || asset.NormalizedPayloadSha256 != expected.NormalizedPayloadSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "multi-prototype manifest provenance mismatch: " + expected.CaptureId);
        }

        static void ValidateSequence(RecognitionSequence sequence, ExpectedPrototype expected, Dictionary<string, object> record)
        {
            if (sequence.SampleId != expected.CaptureId || sequence.Word != expected.Word || sequence.Speed != "PERSONAL" ||
                sequence.Split != "personal_calibration" || sequence.SourceNpzSha256 != expected.SourcePayloadSha256 ||
                sequence.LogicalPayloadSha256 != expected.NormalizedPayloadSha256 || sequence.BinaryPayloadSha256 != expected.EmbeddedChecksumSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "multi-prototype WISRT metadata mismatch: " + expected.CaptureId);
            if (Sha256Coordinates(sequence) != expected.CoordinatePayloadSha256 || Sha256Mask(sequence) != expected.MaskPayloadSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "multi-prototype WISRT payload mismatch: " + expected.CaptureId);
            if (StringValue(record, "word") != expected.Word || StringValue(record, "capture_id") != expected.CaptureId ||
                StringValue(record, "capture_key") != expected.CaptureKey || StringValue(record, "run_id") != expected.RunId ||
                StringValue(record, "source_family") != expected.SourceFamily || Int64Value(record, "frame_count") != expected.FrameCount ||
                StringValue(record, "relative_path") != expected.FileName || Int64Value(record, "size_bytes") != expected.SizeBytes ||
                StringValue(record, "sha256") != expected.FileSha256 || StringValue(record, "embedded_checksum_sha256") != expected.EmbeddedChecksumSha256 ||
                StringValue(record, "source_payload_sha256") != expected.SourcePayloadSha256 ||
                StringValue(record, "coordinate_payload_sha256") != expected.CoordinatePayloadSha256 ||
                StringValue(record, "mask_payload_sha256") != expected.MaskPayloadSha256 ||
                StringValue(record, "normalized_payload_sha256") != expected.NormalizedPayloadSha256)
                throw Failure(PersonalProfileFailureCode.ProvenanceMismatch, "multi-prototype config provenance mismatch: " + expected.CaptureId);
        }

        static byte[] ReadAsset(string path, ManifestAsset asset)
        {
            // path is the canonical absolute result of ResolveSafe; the extended form is
            // applied only to the IO calls and takes no part in containment
            string io = RuntimeLongPath.ForIo(path);
            if (!File.Exists(io)) throw Failure(PersonalProfileFailureCode.AssetMissing, "multi-prototype asset is missing: " + asset.RelativePath);
            if (new FileInfo(io).Length != asset.SizeBytes) throw Failure(PersonalProfileFailureCode.AssetSizeMismatch, "multi-prototype asset size mismatch: " + asset.RelativePath);
            return File.ReadAllBytes(io);
        }

        static void ValidateAssetSha(byte[] bytes, ManifestAsset asset, string expected)
        {
            string actual = Sha256(bytes);
            if (actual != asset.Sha256 || actual != expected)
                throw Failure(PersonalProfileFailureCode.AssetSha256Mismatch, "multi-prototype asset SHA-256 mismatch: " + asset.RelativePath);
        }

        static string ResolveSafe(string root, string relativePath)
        {
            if (string.IsNullOrWhiteSpace(relativePath) || relativePath.IndexOf((char)92) >= 0 || Path.IsPathRooted(relativePath) || relativePath.IndexOf(':') >= 0)
                throw Failure(PersonalProfileFailureCode.UnsafeRelativePath, "unsafe multi-prototype relative path");
            string[] segments = relativePath.Split('/');
            if (segments.Any(segment => segment.Length == 0 || segment == "." || segment == ".."))
                throw Failure(PersonalProfileFailureCode.UnsafeRelativePath, "unsafe multi-prototype relative path");
            string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            string fullPath = Path.GetFullPath(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)));
            if (!fullPath.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase))
                throw Failure(PersonalProfileFailureCode.UnsafeRelativePath, "multi-prototype relative path escapes root");
            return fullPath;
        }

        static void ValidateCountMap(Dictionary<string, object> map)
        {
            ExactKeys(map, PersonalRuntimeProfileLoader.RequiredWords);
            foreach (string word in PersonalRuntimeProfileLoader.RequiredWords)
                if (Int64Value(map, word) != 3) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "prototype count per word mismatch");
        }

        static void ValidateWords(string[] words)
        {
            if (words.Length != 3 || words.Distinct(StringComparer.Ordinal).Count() != 3 ||
                !words.OrderBy(item => item, StringComparer.Ordinal).SequenceEqual(PersonalRuntimeProfileLoader.RequiredWords, StringComparer.Ordinal))
                throw Failure(PersonalProfileFailureCode.WordSetMismatch, "multi-prototype word set mismatch");
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
            return inner == null ? new PersonalProfileLoadException(code, message) : new PersonalProfileLoadException(code, message, inner);
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
            if (!actual.SetEquals(required)) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "JSON key set mismatch");
        }

        static object Required(Dictionary<string, object> value, string name)
        {
            object result;
            if (!value.TryGetValue(name, out result)) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "required JSON field missing: " + name);
            return result;
        }

        static string StringValue(Dictionary<string, object> value, string name)
        {
            string result = Required(value, name) as string;
            if (string.IsNullOrWhiteSpace(result)) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON string: " + name);
            return result;
        }

        static bool BoolValue(Dictionary<string, object> value, string name)
        {
            object result = Required(value, name);
            if (!(result is bool)) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON Boolean: " + name);
            return (bool)result;
        }

        static double NumberValue(Dictionary<string, object> value, string name)
        {
            object result = Required(value, name);
            if (!(result is double)) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON number: " + name);
            return (double)result;
        }

        static long Int64Value(Dictionary<string, object> value, string name)
        {
            double result = NumberValue(value, name);
            if (result < 0 || result > long.MaxValue || result != Math.Floor(result))
                throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON integer: " + name);
            return (long)result;
        }

        static Dictionary<string, object> ObjectValue(Dictionary<string, object> value, string name)
        {
            var result = Required(value, name) as Dictionary<string, object>;
            if (result == null) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON object: " + name);
            return result;
        }

        static List<object> ArrayValue(Dictionary<string, object> value, string name)
        {
            var result = Required(value, name) as List<object>;
            if (result == null) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON array: " + name);
            return result;
        }

        static string[] StringArray(Dictionary<string, object> value, string name)
        {
            List<object> raw = ArrayValue(value, name);
            var result = new string[raw.Count];
            for (int index = 0; index < raw.Count; index++)
            {
                result[index] = raw[index] as string;
                if (string.IsNullOrWhiteSpace(result[index])) throw Failure(PersonalProfileFailureCode.ManifestSchemaMismatch, "invalid JSON string array: " + name);
            }
            return result;
        }
    }

    internal sealed class PersonalMultiPrototypeRuntimeEvaluator : IPersonalRuntimeEvaluator
    {
        readonly PersonalMultiPrototypeRuntimeProfile profile;

        public PersonalMultiPrototypeRuntimeEvaluator(PersonalMultiPrototypeRuntimeProfile profile)
        {
            this.profile = profile ?? throw new ArgumentNullException(nameof(profile));
        }

        public PersonalRuntimeRecognitionResult Recognize(string querySessionId, RecognitionSequence query, string expectedWord)
        {
            if (string.IsNullOrWhiteSpace(querySessionId)) throw new ArgumentException("query/session ID is required", nameof(querySessionId));
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (query.SampleId != querySessionId) throw new ArgumentException("query/session ID mismatch", nameof(querySessionId));
            if (query.Split == "train") throw new ArgumentException("a global train template cannot be a personal runtime query", nameof(query));
            if (!string.IsNullOrEmpty(expectedWord) && !profile.Words.Contains(expectedWord, StringComparer.Ordinal))
                throw new ArgumentException("unsupported expected word", nameof(expectedWord));

            var scores = new Dictionary<string, double>(StringComparer.Ordinal);
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal);
            var distances = new List<PersonalRuntimeTemplateDistance>();
            var winners = new Dictionary<string, RecognitionSequence>(StringComparer.Ordinal);
            foreach (string word in profile.Words)
            {
                RecognitionSequence winner = null;
                double best = double.PositiveInfinity;
                foreach (RecognitionSequence prototype in profile.Prototypes[word])
                {
                    DtwResult dtw = MaskAwareDtw.Compute(query, prototype, RecognitionCore.RelativeBand);
                    distances.Add(new PersonalRuntimeTemplateDistance(word, prototype.SampleId, dtw.Distance, dtw.PathLength));
                    if (winner == null || dtw.Distance < best ||
                        (dtw.Distance == best && string.CompareOrdinal(prototype.SampleId, winner.SampleId) < 0))
                    {
                        winner = prototype;
                        best = dtw.Distance;
                    }
                }
                winners.Add(word, winner);
                scores.Add(word, best);
                accepts.Add(word, best <= profile.Thresholds[word]);
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
                querySessionId, expectedWord, predicted, winners[predicted].SampleId, scores[predicted],
                accepted, expectedAccept, questSuccess, scores, accepts, distances, profile.Warnings);
        }
    }
}
