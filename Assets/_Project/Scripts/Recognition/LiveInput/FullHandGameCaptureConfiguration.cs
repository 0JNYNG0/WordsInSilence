// F21 - the explicit, off-by-default loader for the full-hand game seam.
//
// Nothing here activates by default. With WIS_FULL_HAND_PIPELINE unset or not exactly "1"
// this reads no file, touches no participant and returns null, so the legacy path is
// byte-for-byte unchanged. When it IS enabled, every requirement is explicit: an already
// active demo participant session, a fully qualified bundle manifest path and a fully
// qualified gate file path. There is no default cutoff, no fitted number, no search path,
// no download and no embedded weight anywhere in this file; every threshold comes from the
// operator's gate file and the model identity is the bundle's own payload hash.
//
// The pipeline is constructed common-only with an EMPTY current-participant reference set,
// unless the operator explicitly names an already reviewed F24 reference packet AND its exact
// hash - in which case that packet's members are loaded READ-ONLY for research comparison.
// The existing 16-point personalization candidates are NOT 46-point full-hand references
// and are deliberately neither loaded nor claimed here. F21 contains no enrollment, F24 adds
// none, and a configured packet that cannot be loaded REFUSES instead of quietly falling back.
using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.FullHandRuntime;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// The parsed <c>wis_full_hand_pipeline_gates_v1</c> document. Parsing is pure: it takes
    /// the exact file bytes and the model hash the loaded bundle actually reports, and it
    /// never reads a file, an environment variable or a profile.
    /// </summary>
    public static class FullHandGateDocument
    {
        public const string Schema = "wis_full_hand_pipeline_gates_v1";
        public const int MaxBytes = 65536;
        /// <summary>The only accepted spelling of "this gate has no numeric threshold".</summary>
        public const string NoThreshold = "none";

        [Serializable]
        sealed class GateJson
        {
            public string method;
            public string reject_all;
            public string quality_threshold;
        }

        [Serializable]
        sealed class DocumentJson
        {
            public string schema;
            public string model_payload_sha256;
            public string source_review_sha256;
            public GateJson[] gates;
        }

        /// <summary>
        /// Strict parse. Every field is a JSON STRING on purpose: JsonUtility cannot tell a
        /// missing number from zero and a missing bool from false, so a missing threshold
        /// would silently become a permissive 0.0. With string fields "absent" is the empty
        /// string and is rejected. Anything malformed is rejected; nothing is guessed.
        /// </summary>
        public static bool TryParse(byte[] fileBytes, string expectedModelPayloadSha256,
                                    out FullHandGateConfig config, out string sourceReviewSha256,
                                    out string failureCode, out string failureDetail)
        {
            config = null;
            sourceReviewSha256 = null;
            failureCode = FullHandBridgeCodes.ConfigurationGateInvalid;
            failureDetail = string.Empty;

            if (fileBytes == null || fileBytes.Length == 0 || fileBytes.Length > MaxBytes)
            {
                failureDetail = "the gate document is empty or larger than the accepted limit";
                return false;
            }

            string text;
            try { text = new UTF8Encoding(false, true).GetString(StripBom(fileBytes)); }
            catch (Exception)
            {
                failureDetail = "the gate document is not valid UTF-8";
                return false;
            }

            DocumentJson document;
            try { document = JsonUtility.FromJson<DocumentJson>(text); }
            catch (Exception)
            {
                failureDetail = "the gate document is not valid JSON";
                return false;
            }
            if (document == null)
            {
                failureDetail = "the gate document is not a JSON object";
                return false;
            }
            if (!string.Equals(document.schema, Schema, StringComparison.Ordinal))
            {
                failureDetail = "the gate document declares an unknown schema";
                return false;
            }

            string modelHash = NormalizeSha256(document.model_payload_sha256);
            if (modelHash == null)
            {
                failureDetail = "model_payload_sha256 is missing or is not 64 hex characters";
                return false;
            }
            string reviewHash = NormalizeSha256(document.source_review_sha256);
            if (reviewHash == null)
            {
                failureDetail = "source_review_sha256 is missing or is not 64 hex characters";
                return false;
            }
            string expected = NormalizeSha256(expectedModelPayloadSha256);
            if (expected == null)
            {
                failureDetail = "the loaded bundle did not report a usable payload hash";
                return false;
            }
            if (!string.Equals(modelHash, expected, StringComparison.Ordinal))
            {
                failureCode = FullHandBridgeCodes.ConfigurationModelHashMismatch;
                failureDetail = "the gate document was written for a different model payload";
                return false;
            }

            if (document.gates == null || document.gates.Length != FullHandMethods.All.Length)
            {
                failureDetail = "exactly " + FullHandMethods.All.Length + " gates are required";
                return false;
            }

            // the gate config id is the SHA-256 of the EXACT file bytes, so a one-character
            // edit is a different configuration identity
            string configId = Sha256Hex(fileBytes);
            var gates = new FullHandGate[document.gates.Length];
            for (int index = 0; index < gates.Length; index++)
            {
                GateJson entry = document.gates[index];
                if (entry == null)
                {
                    failureDetail = "a gate entry is absent";
                    return false;
                }
                bool rejectAll;
                if (!TryParseBool(entry.reject_all, out rejectAll))
                {
                    failureDetail = "reject_all must be exactly \"true\" or \"false\"";
                    return false;
                }
                if (string.IsNullOrEmpty(entry.quality_threshold))
                {
                    failureDetail = "quality_threshold is required for every gate";
                    return false;
                }
                string gateId = configId + "_" + entry.method;
                try
                {
                    if (rejectAll)
                    {
                        if (!string.Equals(entry.quality_threshold, NoThreshold,
                                           StringComparison.Ordinal))
                        {
                            failureDetail = "a reject_all gate must declare quality_threshold \"" +
                                            NoThreshold + "\"";
                            return false;
                        }
                        // RequireMethod inside F20 rejects anything but the known three
                        gates[index] = FullHandGate.RejectEverything(gateId, entry.method);
                    }
                    else
                    {
                        double threshold;
                        if (!double.TryParse(entry.quality_threshold, NumberStyles.Float,
                                             CultureInfo.InvariantCulture, out threshold) ||
                            double.IsNaN(threshold) || double.IsInfinity(threshold))
                        {
                            failureDetail = "quality_threshold must be a finite invariant-culture number";
                            return false;
                        }
                        gates[index] = FullHandGate.AtQualityThreshold(gateId, entry.method, threshold);
                    }
                }
                catch (FullHandFormatException exception)
                {
                    failureDetail = exception.Message;
                    return false;
                }
            }

            try { config = new FullHandGateConfig(configId, gates); }
            catch (FullHandFormatException exception)
            {
                failureDetail = exception.Message;
                return false;
            }

            sourceReviewSha256 = reviewHash;
            failureCode = FullHandBridgeCodes.None;
            failureDetail = string.Empty;
            return true;
        }

        static bool TryParseBool(string value, out bool parsed)
        {
            parsed = false;
            if (string.Equals(value, "false", StringComparison.Ordinal)) return true;
            if (string.Equals(value, "true", StringComparison.Ordinal)) { parsed = true; return true; }
            return false;
        }

        static byte[] StripBom(byte[] bytes)
        {
            if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
            {
                var trimmed = new byte[bytes.Length - 3];
                Array.Copy(bytes, 3, trimmed, 0, trimmed.Length);
                return trimmed;
            }
            return bytes;
        }

        /// <summary>Null when absent or not the 64-hex shape. Never invents a hash.</summary>
        public static string NormalizeSha256(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length != 64) return null;
            var text = new StringBuilder(64);
            for (int index = 0; index < 64; index++)
            {
                char c = value[index];
                bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
                if (!hex) return null;
                text.Append(char.ToLowerInvariant(c));
            }
            return text.ToString();
        }

        public static string Sha256Hex(byte[] data)
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

    /// <summary>
    /// The explicit opt-in loader. <see cref="CreateProcessorOrNull"/> returns null - the
    /// legacy path, untouched - unless the feature is deliberately enabled.
    /// </summary>
    public static class FullHandGameCaptureConfiguration
    {
        public const string EnableVariable = "WIS_FULL_HAND_PIPELINE";
        public const string ManifestVariable = "WIS_FULL_HAND_PIPELINE_MANIFEST";
        public const string GatesVariable = "WIS_FULL_HAND_PIPELINE_GATES";
        /// <summary>
        /// F22, opt-in and read ONLY when the pipeline itself is already enabled: a fully
        /// qualified output directory for the local capture-evidence pair. Unset means the
        /// default path - no directory is touched and no capture byte is retained.
        /// </summary>
        public const string TraceDirectoryVariable = "WIS_FULL_HAND_PIPELINE_TRACE";
        /// <summary>
        /// F24, opt-in and read ONLY inside the already enabled branch, after the current
        /// participant, the bundle and the gate document are all valid: the fully qualified
        /// path of an explicitly reviewed current-participant reference packet. Unset means
        /// the default - the EMPTY set, and no file is read.
        /// </summary>
        public const string ReferencesVariable = "WIS_FULL_HAND_PIPELINE_REFERENCES";
        /// <summary>
        /// The SHA-256 of the EXACT reviewed packet bytes, required whenever the packet path
        /// is set. It is the trust boundary for the reviewed selection and its labels; it does
        /// not certify that any capture is a well formed sign.
        /// </summary>
        public const string ReferencesSha256Variable =
            "WIS_FULL_HAND_PIPELINE_REFERENCES_SHA256";
        public const string EnabledValue = "1";

        /// <summary>True only for the exact opt-in value. Reads no file.</summary>
        public static bool IsEnabled()
        {
            return string.Equals(ReadVariable(EnableVariable), EnabledValue, StringComparison.Ordinal);
        }

        /// <summary>
        /// Null when disabled: no file is read and no participant is initialized on that
        /// path. When ENABLED, this never returns null - an invalid configuration returns a
        /// refusing processor so the operator sees the typed failure on every capture.
        /// </summary>
        public static IFullHandGameCaptureProcessor CreateProcessorOrNull()
        {
            if (!IsEnabled()) return null;
            try { return CreateEnabledProcessor(); }
            catch (Exception exception)
            {
                return new FullHandGameCaptureRefusingProcessor(
                    FullHandBridgeCodes.ConfigurationRefused, exception.GetType().Name);
            }
        }

        static IFullHandGameCaptureProcessor CreateEnabledProcessor()
        {
            string participantId = CurrentParticipantIdOrNull();
            if (string.IsNullOrEmpty(participantId))
                return Refuse(FullHandBridgeCodes.ConfigurationParticipantMissing,
                    "the full-hand pipeline requires an already active demo participant session");

            string manifestPath = ReadVariable(ManifestVariable);
            string gatesPath = ReadVariable(GatesVariable);
            if (!IsFullyQualifiedExistingFile(manifestPath))
                return Refuse(FullHandBridgeCodes.ConfigurationPathMissing,
                    "the bundle manifest path must be a fully qualified existing file");
            if (!IsFullyQualifiedExistingFile(gatesPath))
                return Refuse(FullHandBridgeCodes.ConfigurationPathMissing,
                    "the gate document path must be a fully qualified existing file");

            byte[] gateBytes;
            try
            {
                var info = new FileInfo(gatesPath);
                if (info.Length > FullHandGateDocument.MaxBytes)
                    return Refuse(FullHandBridgeCodes.ConfigurationGateUnreadable,
                        "the gate document is larger than the accepted limit");
                gateBytes = File.ReadAllBytes(gatesPath);
            }
            catch (Exception)
            {
                return Refuse(FullHandBridgeCodes.ConfigurationGateUnreadable,
                    "the gate document could not be read");
            }

            FullHandRuntimeBundle bundle;
            try { bundle = FullHandRuntimeBundle.Load(manifestPath); }
            catch (Exception)
            {
                return Refuse(FullHandBridgeCodes.ConfigurationModelUnreadable,
                    "the full-hand bundle could not be loaded");
            }
            if (bundle == null)
                return Refuse(FullHandBridgeCodes.ConfigurationModelUnreadable,
                    "the full-hand bundle could not be loaded");

            FullHandGateConfig gates;
            string reviewHash, failureCode, failureDetail;
            if (!FullHandGateDocument.TryParse(gateBytes, bundle.PayloadSha256, out gates,
                                               out reviewHash, out failureCode, out failureDetail))
                return Refuse(failureCode, failureDetail);

            // F24: EMPTY unless the operator named a reviewed packet AND its exact hash. This
            // mines no directory, imports no legacy or previous-participant profile, enrolls
            // nothing, changes no threshold or model, and claims no personalized improvement.
            // The result is a frozen per-controller snapshot: changing it requires deliberate
            // reconstruction, never a mid-evaluation edit.
            string referencesPath = ReadVariable(ReferencesVariable);
            string referencesSha256 = ReadVariable(ReferencesSha256Variable);
            FullHandParticipantReferenceSet references;
            try
            {
                references = FullHandParticipantReferencePacket.LoadOrEmpty(
                    referencesPath, referencesSha256, participantId, bundle.PayloadSha256,
                    gates.ConfigId, bundle.CopyVocabulary());
            }
            catch (FullHandFormatException exception)
            {
                // a configured packet that cannot be loaded is a REFUSAL, never a silent
                // common-only fallback that would still look like a successful research run
                return Refuse(exception.Code,
                    "the configured current-participant reference packet was refused");
            }
            catch (Exception exception)
            {
                return Refuse(FullHandBridgeCodes.ConfigurationRefused,
                    exception.GetType().Name);
            }

            try
            {
                var pipeline = new FullHandRecognitionPipeline(bundle, gates, participantId,
                    references);

                // F22: null unless the operator configured a trace directory. No writer means
                // no retention, so the disabled path keeps no capture data after scoring.
                FullHandCaptureTraceWriter writer =
                    FullHandCaptureTraceWriter.CreateOrNull(ReadVariable(TraceDirectoryVariable));
                // F28: the explicit gameplay opt-in also needs these already built, already
                // hashed bytes, because an accepted capture may be collected as a local
                // practice record. It changes no math, no gate and no decision. With gameplay
                // OFF this is false and the default path still retains nothing.
                bool retainForRecordCollection = FullHandGameplayConfiguration.IsEnabled();
                var processor = new FullHandGameCaptureProcessor(pipeline,
                    CurrentParticipantIdOrNull, writer != null || retainForRecordCollection);
                if (writer == null) return processor;
                return new FullHandGameCaptureTracingProcessor(processor, writer);
            }
            catch (Exception exception)
            {
                return Refuse(FullHandBridgeCodes.ConfigurationRefused, exception.GetType().Name);
            }
        }

        static IFullHandGameCaptureProcessor Refuse(string code, string detail)
        {
            return new FullHandGameCaptureRefusingProcessor(code, detail);
        }

        /// <summary>
        /// The single identity source: the already active demo participant session. Never a
        /// legacy fallback identity, and never a newly created one.
        /// </summary>
        public static string CurrentParticipantIdOrNull()
        {
            try
            {
                if (!DemoParticipantSession.IsActive) return null;
                DemoParticipantIdentity identity = DemoParticipantSession.Current;
                return identity == null ? null : identity.LearnerId;
            }
            catch (Exception) { return null; }
        }

        static bool IsFullyQualifiedExistingFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try
            {
                if (!Path.IsPathFullyQualified(path)) return false;
                foreach (string segment in path.Split('/', '\\'))
                    if (segment == "." || segment == "..") return false;
                return File.Exists(path);
            }
            catch (Exception) { return false; }
        }

        static string ReadVariable(string name)
        {
            try { return Environment.GetEnvironmentVariable(name); }
            catch (Exception) { return null; }
        }
    }
}
