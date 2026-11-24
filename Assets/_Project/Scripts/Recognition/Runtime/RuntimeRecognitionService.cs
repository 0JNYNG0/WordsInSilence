using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Evaluation;

namespace WordsInSilence.Recognition.Runtime
{
    public sealed class RuntimeTemplateDistance
    {
        public string PrototypeId { get; }
        public string PrototypeWord { get; }
        public string PrototypeSpeed { get; }
        public double Distance { get; }
        public int PathLength { get; }

        public RuntimeTemplateDistance(string prototypeId, string prototypeWord, string prototypeSpeed, double distance, int pathLength)
        {
            PrototypeId = prototypeId;
            PrototypeWord = prototypeWord;
            PrototypeSpeed = prototypeSpeed;
            Distance = distance;
            PathLength = pathLength;
        }
    }

    /// <summary>Immutable runtime-facing result after RecognitionCore and the frozen gate.</summary>
    public sealed class RuntimeRecognitionResult
    {
        public string QuerySessionId { get; }
        public string PredictedWord { get; }
        public string WinningPrototypeId { get; }
        public double GreetingDistance { get; }
        public double LoveDistance { get; }
        public double ThankYouDistance { get; }
        public double BestDistance { get; }
        public double Threshold { get; }
        public bool Accepted { get; }
        public IReadOnlyList<RuntimeTemplateDistance> TemplateDistances { get; }

        public RuntimeRecognitionResult(
            string querySessionId,
            string predictedWord,
            string winningPrototypeId,
            double greetingDistance,
            double loveDistance,
            double thankYouDistance,
            double bestDistance,
            double threshold,
            bool accepted,
            IReadOnlyList<RuntimeTemplateDistance> templateDistances)
        {
            QuerySessionId = querySessionId;
            PredictedWord = predictedWord;
            WinningPrototypeId = winningPrototypeId;
            GreetingDistance = greetingDistance;
            LoveDistance = loveDistance;
            ThankYouDistance = thankYouDistance;
            BestDistance = bestDistance;
            Threshold = threshold;
            Accepted = accepted;
            TemplateDistances = templateDistances ?? throw new ArgumentNullException(nameof(templateDistances));
        }
    }

    /// <summary>Pure C# immutable runtime boundary over the approved templates, core and distance gate.</summary>
    public sealed class RuntimeRecognitionService
    {
        public const int FrozenThresholdFileSize = 4590;
        public const string FrozenThresholdFileSha256 = "2697bb1d0c301850d2abda41351519c66d0eb8e27841263469175eb3fa7e99e3";
        public const string FrozenThresholdCanonicalPayloadSha256 = "8600a9529c82a4d30dea984515bc56f6ade74c12331a3df3fcf59f3ffcf7f7c0";

        sealed class ApprovedTemplate
        {
            public string FileName { get; }
            public string SampleId { get; }
            public string Sha256 { get; }

            public ApprovedTemplate(string fileName, string sampleId, string sha256)
            {
                FileName = fileName;
                SampleId = sampleId;
                Sha256 = sha256;
            }
        }

        static readonly ApprovedTemplate[] ApprovedTemplates =
        {
            new ApprovedTemplate("template_GREETING__FAST.wisrt", "S001_P001_GREETING_FAST_R02", "a9111bd2733cb8a2d2c5984309607979cee49c2516346668228e4fb51cf88277"),
            new ApprovedTemplate("template_GREETING__NORMAL.wisrt", "S001_P001_GREETING_NORMAL_R03", "8a2859403e0db736d213f59c82bd9d9a3f7a44dc748b6c65dca5869bc4cfa4b9"),
            new ApprovedTemplate("template_GREETING__SLOW.wisrt", "S001_P001_GREETING_SLOW_R01", "0646436e35e20d13e3c00fd0aef85c7a77829d3729eb7a08e2c9ffa3dc989680"),
            new ApprovedTemplate("template_LOVE__FAST.wisrt", "S001_P001_LOVE_FAST_R01", "1ab0ccee87c749cb961f6d8a8135062f14c8affc8ec15b40472240e3e936967c"),
            new ApprovedTemplate("template_LOVE__NORMAL.wisrt", "S001_P001_LOVE_NORMAL_R02", "74b42959641fd1ce182f4f72427c02307f14ddc564e06e6f24044694bca5a3c3"),
            new ApprovedTemplate("template_LOVE__SLOW.wisrt", "S001_P001_LOVE_SLOW_R01", "509cfb87126d18285745085810fc4a5afcc82886a8869d02f375166f71819a28"),
            new ApprovedTemplate("template_THANK_YOU__FAST.wisrt", "S001_P001_THANK_YOU_FAST_R02", "e53455dff4448865b0df7d9d859888213a31e33738512b7ed7af0d16230e3e6f"),
            new ApprovedTemplate("template_THANK_YOU__NORMAL.wisrt", "S001_P001_THANK_YOU_NORMAL_R05", "7da43c9f770ac79e1a67eb487469cef35467b72f4ab3abbb41faa26f0396c82e"),
            new ApprovedTemplate("template_THANK_YOU__SLOW.wisrt", "S001_P001_THANK_YOU_SLOW_R01", "b83309ac244a8f92fd83ac143da9e2307768125fe3379fd0ff9345bf6b5bb26f"),
        };

        readonly RecognitionCore core;
        readonly FrozenDistanceGateConfig gateConfig;
        readonly IReadOnlyList<string> templateIds;

        public int TemplateCount => templateIds.Count;
        public IReadOnlyList<string> TemplateIds => templateIds;
        public double Threshold => gateConfig.Threshold;
        public string ThresholdCanonicalPayloadSha256 => gateConfig.CanonicalPayloadSha256;

        RuntimeRecognitionService(RecognitionTemplateLibrary library, FrozenDistanceGateConfig gateConfig)
        {
            core = new RecognitionCore(library ?? throw new ArgumentNullException(nameof(library)));
            this.gateConfig = gateConfig ?? throw new ArgumentNullException(nameof(gateConfig));
            templateIds = Array.AsReadOnly(library.Templates.Select(item => item.SampleId).ToArray());
        }

        public static RuntimeRecognitionService Load(string templateDirectory, string thresholdConfigPath)
        {
            if (string.IsNullOrWhiteSpace(templateDirectory)) throw new ArgumentException("template directory is required", nameof(templateDirectory));
            if (string.IsNullOrWhiteSpace(thresholdConfigPath)) throw new ArgumentException("threshold config path is required", nameof(thresholdConfigPath));
            if (!Directory.Exists(templateDirectory)) throw new DirectoryNotFoundException(templateDirectory);
            if (!File.Exists(thresholdConfigPath)) throw new FileNotFoundException("runtime threshold config is missing", thresholdConfigPath);

            RecognitionTemplateLibrary library = RecognitionTemplateLibrary.LoadDirectory(templateDirectory);
            string[] actualPaths = Directory.GetFiles(templateDirectory, "*.wisrt", SearchOption.TopDirectoryOnly);
            if (actualPaths.Length != ApprovedTemplates.Length)
                throw new InvalidDataException($"exactly nine approved template binaries are required; found {actualPaths.Length}");
            var loadedById = library.Templates.ToDictionary(item => item.SampleId, StringComparer.Ordinal);
            foreach (ApprovedTemplate expected in ApprovedTemplates)
            {
                string path = Path.Combine(templateDirectory, expected.FileName);
                if (!File.Exists(path)) throw new InvalidDataException($"approved template file is missing: {expected.FileName}");
                if (!string.Equals(Sha256File(path), expected.Sha256, StringComparison.Ordinal))
                    throw new InvalidDataException($"approved template SHA-256 mismatch: {expected.FileName}");
                if (!loadedById.ContainsKey(expected.SampleId))
                    throw new InvalidDataException($"approved template sample ID is missing: {expected.SampleId}");
            }

            byte[] thresholdBytes = File.ReadAllBytes(thresholdConfigPath);
            if (thresholdBytes.Length != FrozenThresholdFileSize)
                throw new InvalidDataException("runtime threshold config size mismatch");
            if (!string.Equals(Sha256Bytes(thresholdBytes), FrozenThresholdFileSha256, StringComparison.Ordinal))
                throw new InvalidDataException("runtime threshold config file SHA-256 mismatch");
            FrozenDistanceGateConfig config = FrozenDistanceGateConfigLoader.Load(thresholdBytes, Array.Empty<string>());
            if (!string.Equals(config.CanonicalPayloadSha256, FrozenThresholdCanonicalPayloadSha256, StringComparison.Ordinal))
                throw new InvalidDataException("runtime threshold canonical payload SHA-256 mismatch");
            return new RuntimeRecognitionService(library, config);
        }

        public RuntimeRecognitionResult Recognize(string querySessionId, RecognitionSequence query)
        {
            if (string.IsNullOrWhiteSpace(querySessionId)) throw new ArgumentException("query/session ID is required", nameof(querySessionId));
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (!string.Equals(querySessionId, query.SampleId, StringComparison.Ordinal))
                throw new ArgumentException("query/session ID does not match the finalized sequence", nameof(querySessionId));
            if (string.Equals(query.Split, "train", StringComparison.Ordinal))
                throw new ArgumentException("a train template cannot be submitted as a runtime query", nameof(query));

            RecognitionResult recognition = core.Recognize(query);
            DistanceGateDecision decision = DistanceGatePolicy.Evaluate(recognition, gateConfig.Threshold);
            var words = recognition.WordDistances.ToDictionary(item => item.Word, StringComparer.Ordinal);
            if (!words.ContainsKey("GREETING") || !words.ContainsKey("LOVE") || !words.ContainsKey("THANK_YOU") || words.Count != 3)
                throw new InvalidDataException("RecognitionCore returned an invalid word-distance set");
            var templateDistances = recognition.TemplateDistances
                .Select(item => new RuntimeTemplateDistance(
                    item.Prototype.SampleId,
                    item.Prototype.Word,
                    item.Prototype.Speed,
                    item.Distance,
                    item.PathLength))
                .ToArray();
            return new RuntimeRecognitionResult(
                querySessionId,
                decision.PredictedWord,
                decision.WinningPrototype.SampleId,
                words["GREETING"].Distance,
                words["LOVE"].Distance,
                words["THANK_YOU"].Distance,
                decision.BestWordDistance,
                gateConfig.Threshold,
                decision.Accepted,
                Array.AsReadOnly(templateDistances));
        }

        static string Sha256File(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var sha = SHA256.Create())
                return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(stream));
        }

        static string Sha256Bytes(byte[] bytes)
        {
            using (var sha = SHA256.Create())
                return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }
    }
}
