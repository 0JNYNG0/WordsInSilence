using System;
using System.IO;
using System.Security.Cryptography;

namespace WordsInSilence.Recognition.Common
{
    /// <summary>
    /// The frozen Step 10O v1.6 common-2D model, loaded fail-closed from immutable
    /// little-endian float64 bytes.
    ///
    /// Nothing here trains, refits, searches or writes. Every expected SHA-256, shape
    /// and the class order are pinned in source, so a swapped or edited artifact
    /// cannot load. The delivered common_model_manifest.json is documentation for
    /// external review; load correctness never depends on parsing it.
    /// </summary>
    public sealed class FrozenCommonSignModel
    {
        public const string ArtifactVersion = "common_2d_v1_6";
        public const string SourceDeliverySha256 =
            "8bbc25b859ac5454db93415633ecb3efb630f9e7f97ae532753288241f333d21";
        public const long SourceDeliverySize = 22403722L;

        public const string ClassGreeting = "GREETING";
        public const string ClassThankYou = "THANK_YOU";

        public const int LandmarkCount = 16;
        public const int ChannelCount = 3;
        public const int ResampleT = 32;
        public const int FeatureCount = 1536;
        public const int PcaComponents = 4;

        public const string ScalerMeanFile = "scaler_mean.bin";
        public const string ScalerScaleFile = "scaler_scale.bin";
        public const string PcaMeanFile = "pca_mean.bin";
        public const string PcaComponentsFile = "pca_components.bin";
        public const string LrCoefFile = "lr_coef.bin";
        public const string LrInterceptFile = "lr_intercept.bin";
        public const string ManifestFile = "common_model_manifest.json";

        // Pinned identity of every array actually installed for the runtime, and of
        // their fixed-order concatenation. A swapped or edited artifact cannot load.
        public const string ScalerMeanSha256 =
            "f7a2b20cbf42063756137fb6b31a62ffba64ebe1d8453ec8b5d7c43d1b2ebba8";
        public const string ScalerScaleSha256 =
            "7008d23b32ae3c5527fa901567815aa9d03781743e820cd88a722c9ace7259ef";
        public const string PcaMeanSha256 =
            "ba0e081eff54c6f7b78fec845221a406b0e7e1c0c0a16335a3541550ccc4bcdb";
        public const string PcaComponentsSha256 =
            "7a1f335f0de8fd9b12f1ac3ee3d862ed4327cc6d32ea7d0edfaa6792d4c6587a";
        public const string LrCoefSha256 =
            "22a8ae9545332f1ddaaae37a3f5145a9df1f8becc8541f651b7221326bb68080";
        public const string LrInterceptSha256 =
            "b12c29bdbc34ddf005a98e3bcb6038624536bd22ae96837f49587f92c5a840c6";
        public const string ExpectedCombinedSha256 =
            "4bb14539835ea64cdd6e77a4672a51ba9caa9da21dd57eaec6f7516fa7379158";

        readonly double[] scalerMean;
        readonly double[] scalerScale;
        readonly double[] pcaMean;
        readonly double[] pcaComponents;   // [PcaComponents, FeatureCount] C-order
        readonly double[] lrCoef;          // [PcaComponents]
        readonly double lrIntercept;

        public string ArtifactSha256 { get; }
        public string SourceDirectory { get; }

        FrozenCommonSignModel(
            string sourceDirectory,
            string artifactSha256,
            double[] scalerMean,
            double[] scalerScale,
            double[] pcaMean,
            double[] pcaComponents,
            double[] lrCoef,
            double lrIntercept)
        {
            SourceDirectory = sourceDirectory;
            ArtifactSha256 = artifactSha256;
            this.scalerMean = scalerMean;
            this.scalerScale = scalerScale;
            this.pcaMean = pcaMean;
            this.pcaComponents = pcaComponents;
            this.lrCoef = lrCoef;
            this.lrIntercept = lrIntercept;
        }

        /// <summary>
        /// Loads and verifies the model. Returns null and sets <paramref name="failure"/>
        /// on any integrity problem; never throws for a bad artifact and never falls
        /// through to a usable model.
        /// </summary>
        public static FrozenCommonSignModel TryLoad(
            string directory,
            out CommonSignFailureReason failure,
            out string detail)
        {
            failure = CommonSignFailureReason.ModelUnavailable;
            detail = string.Empty;
            if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            {
                detail = "model directory not found: " + (directory ?? "<null>");
                return null;
            }

            double[] mean, scale, pMean, pComp, coef, intercept;
            if (!TryReadArray(directory, ScalerMeanFile, FeatureCount, ScalerMeanSha256, out mean, out detail) ||
                !TryReadArray(directory, ScalerScaleFile, FeatureCount, ScalerScaleSha256, out scale, out detail) ||
                !TryReadArray(directory, PcaMeanFile, FeatureCount, PcaMeanSha256, out pMean, out detail) ||
                !TryReadArray(directory, PcaComponentsFile, PcaComponents * FeatureCount, PcaComponentsSha256, out pComp, out detail) ||
                !TryReadArray(directory, LrCoefFile, PcaComponents, LrCoefSha256, out coef, out detail) ||
                !TryReadArray(directory, LrInterceptFile, 1, LrInterceptSha256, out intercept, out detail))
            {
                failure = CommonSignFailureReason.ModelIntegrityMismatch;
                return null;
            }

            for (int i = 0; i < FeatureCount; i++)
            {
                if (scale[i] == 0.0 || double.IsNaN(scale[i]) || double.IsInfinity(scale[i]))
                {
                    failure = CommonSignFailureReason.ModelIntegrityMismatch;
                    detail = "scaler scale must be finite and non-zero at index " + i;
                    return null;
                }
            }

            string combined = CombinedSha256(directory);
            if (!string.Equals(combined, ExpectedCombinedSha256, StringComparison.Ordinal))
            {
                failure = CommonSignFailureReason.ModelIntegrityMismatch;
                detail = "combined artifact digest mismatch: expected " +
                    ExpectedCombinedSha256 + ", got " + combined;
                return null;
            }

            failure = CommonSignFailureReason.None;
            detail = string.Empty;
            return new FrozenCommonSignModel(
                directory, combined, mean, scale, pMean, pComp, coef, intercept[0]);
        }

        static string Sha256Hex(byte[] bytes)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(bytes);
                var text = new System.Text.StringBuilder(digest.Length * 2);
                for (int i = 0; i < digest.Length; i++) text.Append(digest[i].ToString("x2"));
                return text.ToString();
            }
        }

        /// <summary>Order-independent-free, order-FIXED digest over the six model files.</summary>
        public static string CombinedSha256(string directory)
        {
            string[] files =
            {
                ScalerMeanFile, ScalerScaleFile, PcaMeanFile,
                PcaComponentsFile, LrCoefFile, LrInterceptFile,
            };
            using (SHA256 sha = SHA256.Create())
            {
                using (var buffer = new MemoryStream())
                {
                    for (int i = 0; i < files.Length; i++)
                    {
                        byte[] bytes = File.ReadAllBytes(Path.Combine(directory, files[i]));
                        buffer.Write(bytes, 0, bytes.Length);
                    }
                    byte[] digest = sha.ComputeHash(buffer.ToArray());
                    var text = new System.Text.StringBuilder(digest.Length * 2);
                    for (int i = 0; i < digest.Length; i++) text.Append(digest[i].ToString("x2"));
                    return text.ToString();
                }
            }
        }

        static bool TryReadArray(
            string directory, string fileName, int expectedElements, string expectedSha256,
            out double[] values, out string detail)
        {
            values = null;
            detail = string.Empty;
            string path = Path.Combine(directory, fileName);
            if (!File.Exists(path))
            {
                detail = "missing model array: " + fileName;
                return false;
            }
            byte[] bytes = File.ReadAllBytes(path);
            int expectedBytes = expectedElements * sizeof(double);
            if (bytes.Length != expectedBytes)
            {
                detail = fileName + ": expected " + expectedBytes + " bytes, got " + bytes.Length;
                return false;
            }
            string actualSha = Sha256Hex(bytes);
            if (!string.Equals(actualSha, expectedSha256, StringComparison.Ordinal))
            {
                detail = fileName + ": SHA-256 mismatch, expected " + expectedSha256 +
                    ", got " + actualSha;
                return false;
            }
            var result = new double[expectedElements];
            for (int i = 0; i < expectedElements; i++)
            {
                result[i] = ReadLittleEndianDouble(bytes, i * sizeof(double));
                if (double.IsNaN(result[i]) || double.IsInfinity(result[i]))
                {
                    detail = fileName + ": non-finite value at element " + i;
                    return false;
                }
            }
            values = result;
            return true;
        }

        /// <summary>Explicit little-endian float64 read, independent of host endianness.</summary>
        internal static double ReadLittleEndianDouble(byte[] bytes, int offset)
        {
            ulong bits = 0;
            for (int i = 7; i >= 0; i--) bits = (bits << 8) | bytes[offset + i];
            return BitConverter.Int64BitsToDouble(unchecked((long)bits));
        }

        /// <summary>
        /// Frozen forward pass: standardize, project, decide. Returns the raw decision
        /// value; probability mapping is applied by the caller exactly as the frozen
        /// binary liblinear convention requires.
        /// </summary>
        public double Decide(double[] features)
        {
            return DecideInternal(features, null, null, null);
        }

        /// <summary>
        /// The single arithmetic path. The three optional sinks, when non-null, receive
        /// copies of values this path has already produced; they are written after the
        /// value exists and never participate in the calculation, so the decision is
        /// bit-identical whether or not an observer is attached.
        /// </summary>
        double DecideInternal(
            double[] features,
            double[] standardizedSink,
            double[] projectionSink,
            double[] contributionSink)
        {
            if (features == null) throw new ArgumentNullException(nameof(features));
            if (features.Length != FeatureCount)
                throw new ArgumentException("feature vector must contain exactly " +
                    FeatureCount + " values", nameof(features));

            var standardized = new double[FeatureCount];
            for (int i = 0; i < FeatureCount; i++)
                standardized[i] = (features[i] - scalerMean[i]) / scalerScale[i];

            double decision = lrIntercept;
            for (int k = 0; k < PcaComponents; k++)
            {
                double projection = 0.0;
                int rowOffset = k * FeatureCount;
                for (int i = 0; i < FeatureCount; i++)
                    projection += (standardized[i] - pcaMean[i]) * pcaComponents[rowOffset + i];
                double contribution = projection * lrCoef[k];
                decision += contribution;
                if (projectionSink != null) projectionSink[k] = projection;
                if (contributionSink != null) contributionSink[k] = contribution;
            }
            if (standardizedSink != null)
                Array.Copy(standardized, 0, standardizedSink, 0, FeatureCount);
            return decision;
        }

#if UNITY_EDITOR
        /// <summary>
        /// Editor-only observation entry point for the Step 10P v1.0f2 diagnostic trace.
        /// It runs the same <see cref="DecideInternal"/> body as <see cref="Decide"/>;
        /// no shadow forward pass exists. The caller owns the sink arrays, so no live
        /// model array is ever handed out.
        /// </summary>
        public double DecideWithTrace(
            double[] features,
            double[] standardizedSink,
            double[] projectionSink,
            double[] contributionSink,
            out double intercept)
        {
            intercept = lrIntercept;
            return DecideInternal(features, standardizedSink, projectionSink, contributionSink);
        }
#endif

        /// <summary>Exposed for parity evidence only; returns a copy.</summary>
        public double[] Project(double[] features)
        {
            if (features == null) throw new ArgumentNullException(nameof(features));
            if (features.Length != FeatureCount)
                throw new ArgumentException("feature vector length", nameof(features));
            var projections = new double[PcaComponents];
            var standardized = new double[FeatureCount];
            for (int i = 0; i < FeatureCount; i++)
                standardized[i] = (features[i] - scalerMean[i]) / scalerScale[i];
            for (int k = 0; k < PcaComponents; k++)
            {
                double projection = 0.0;
                int rowOffset = k * FeatureCount;
                for (int i = 0; i < FeatureCount; i++)
                    projection += (standardized[i] - pcaMean[i]) * pcaComponents[rowOffset + i];
                projections[k] = projection;
            }
            return projections;
        }

        /// <summary>
        /// Frozen binary liblinear probability convention:
        /// p(class_order[1]) = logistic(decision); p(class_order[0]) = 1 - that.
        /// </summary>
        public static void ToProbabilities(
            double decision, out double greetingProbability, out double thankYouProbability)
        {
            double positive = 1.0 / (1.0 + Math.Exp(-decision));
            thankYouProbability = positive;
            greetingProbability = 1.0 - positive;
        }

        public static bool IsSupportedTarget(string word)
        {
            return string.Equals(word, ClassGreeting, StringComparison.Ordinal) ||
                   string.Equals(word, ClassThankYou, StringComparison.Ordinal);
        }
    }
}
