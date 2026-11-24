#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace WordsInSilence.Recognition.Diagnostics
{
    /// <summary>
    /// Step 10P v1.0f2 common-2D diagnostic export.
    ///
    /// Editor only. Serializes one finalized trace into a dedicated project-relative
    /// validation root: binary little-endian float64 arrays, canonically packed bool
    /// blocks and deterministic UTF-8 JSON without BOM, all listed once by a canonical
    /// manifest that excludes its own digest.
    ///
    /// It is called only after the classification and the attempt result already exist.
    /// It never throws into the capture path: every failure becomes a typed, bounded
    /// result. It never overwrites, merges with or truncates an existing final or
    /// temporary export; content is staged under a temporary directory, validated, and
    /// only then atomically renamed to the export identity directory.
    /// </summary>
    public enum Common2DExportFaultInjection
    {
        None,
        SerializationThrows,
        WriteThrows,
    }

    /// <summary>
    /// The eight independent ways a published member can violate the canonical schema.
    /// Each one is injected on its own so a single defect proves a single rejection; no
    /// case relies on another field being wrong at the same time.
    /// </summary>
    public enum Common2DMemberSchemaMutation
    {
        None,
        Name,
        Order,
        Group,
        Dtype,
        Shape,
        ElementCount,
        ByteCount,
        PadBit,
    }

    public sealed class Common2DDiagnosticExportResult
    {
        public Common2DTraceStatus Status { get; internal set; }
        public Common2DTraceFailureReason FailureReason { get; internal set; }
        public string ExportId { get; internal set; }
        public string PayloadSha256 { get; internal set; }
        public string ArrayShapes { get; internal set; }
        public string ArraySha256s { get; internal set; }
        public string Marker { get; internal set; }
        public int MemberCount { get; internal set; }

        public bool Committed
        {
            get { return Status == Common2DTraceStatus.Committed; }
        }

        internal Common2DDiagnosticExportResult()
        {
            Status = Common2DTraceStatus.Failed;
            FailureReason = Common2DTraceFailureReason.None;
            ExportId = string.Empty;
            PayloadSha256 = string.Empty;
            ArrayShapes = string.Empty;
            ArraySha256s = string.Empty;
            Marker = string.Empty;
        }
    }

    /// <summary>One serialized payload member, already materialized as bytes.</summary>
    sealed class ExportMember
    {
        public string Name;
        public string Dtype;
        public string Shape;
        public long ElementCount;
        public byte[] Bytes;
        public string Sha256;
        public string Group;
    }

    public static class Common2DDiagnosticTraceExporter
    {
        public const string ExportIdPrefix = "WIS_COMMON2D_TRACE_V1";

        /// <summary>
        /// The ONE project-relative validation root the developer route may write to.
        /// It is not configurable: no serialized field, argument or environment value can
        /// redirect a developer export anywhere else.
        /// </summary>
        public const string DeveloperExportRoot = "Validation/Common2DDiagnosticExports/v1";

        /// <summary>Directory names a diagnostic export may never resolve into.</summary>
        public static readonly string[] ProtectedRootNames =
        {
            "Assets", "Packages", "ProjectSettings", "Library",
            "PersonalizationCandidates", "Profiles", "PersonalData",
            "Recordings", "TrainingSamples",
        };
        public const string ManifestName = "payload_manifest.json";
        public const string TemporaryPrefix = ".staging-";
        public const ulong CanonicalQuietNaNBits = 0x7ff8000000000000UL;
        public const string JsonNaNSentinel = "NaN";

        /// <summary>
        /// The exact published member schema: twenty-four members in canonical order
        /// with their fixed name, group and dtype. Shapes whose extent depends on the
        /// capture are written as a template; every other shape is literal.
        /// </summary>
        public static readonly string[][] CanonicalMemberSchema =
        {
            new[] {"raw_timestamps.bin", "raw_observation_sequence", "<f8", "[N]"},
            new[] {"raw_pose_coordinates.bin", "raw_observation_sequence", "<f8", "[N,33,3]"},
            new[] {"raw_left_hand_coordinates.bin", "raw_observation_sequence", "<f8", "[N,21,3]"},
            new[] {"raw_right_hand_coordinates.bin", "raw_observation_sequence", "<f8", "[N,21,3]"},
            new[] {"raw_pose_valid_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,33]"},
            new[] {"raw_left_hand_valid_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_right_hand_valid_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_pose_reappearance_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,33]"},
            new[] {"raw_left_hand_reappearance_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_right_hand_reappearance_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_pose_strong_depth_boundary_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,33]"},
            new[] {"raw_left_hand_strong_depth_boundary_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_right_hand_strong_depth_boundary_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_convention.json", "raw_observation_sequence", "utf8_json", "[]"},
            new[] {"raw_geometry.json", "raw_observation_sequence", "utf8_json", "[]"},
            new[] {"normalized_coordinates.bin", "normalized_common2d_coordinates", "<f8", "[T,16,3]"},
            new[] {"input_mask.bin", "common2d_input_mask", "packed_bool_lsb_first", "[T,16]"},
            new[] {"resampled_coordinates.bin", "resampled_common2d_coordinates", "<f8", "[32,16,2]"},
            new[] {"resampled_mask.bin", "resampled_common2d_mask", "packed_bool_lsb_first", "[32,16]"},
            new[] {"feature_vector.bin", "feature_vector", "<f8", "[1536]"},
            new[] {"standardized_feature_vector.bin", "standardized_feature_vector", "<f8", "[1536]"},
            new[] {"pca_projection.bin", "pca_projection", "<f8", "[4]"},
            new[] {"lr_component_contributions.bin", "lr_component_contributions", "<f8", "[4]"},
            new[] {"lr_scalars.bin", "lr_intercept_decision_probabilities", "<f8", "[4]"},
        };

        public const int CanonicalMemberCount = 24;

        /// <summary>The ten canonical v1.0f1 payload groups, in fixed order.</summary>
        public static readonly string[] CanonicalPayloadGroups =
        {
            "raw_observation_sequence",
            "normalized_common2d_coordinates",
            "common2d_input_mask",
            "resampled_common2d_coordinates",
            "resampled_common2d_mask",
            "feature_vector",
            "standardized_feature_vector",
            "pca_projection",
            "lr_component_contributions",
            "lr_intercept_decision_probabilities",
        };

        /// <summary>Test-only fault seam. Default None; never set by production code.</summary>
        public static Common2DExportFaultInjection FaultInjection { get; set; }

        /// <summary>
        /// Test-only member-schema defect seam. Default None; never set by production
        /// code. It mutates one built member immediately before validation so a single
        /// name, order, group, dtype, shape, element-count, byte-count or pad-bit defect
        /// is rejected on its own, before any root or staging path exists.
        /// </summary>
        public static Common2DMemberSchemaMutation SchemaMutation { get; set; }

        /// <summary>Which member the schema mutation is applied to. Default 0.</summary>
        public static int SchemaMutationMemberIndex { get; set; }

        /// <summary>
        /// Applies exactly one schema defect to the built member list. The mutation is
        /// applied after serialization and before validation, so it cannot be swallowed
        /// as a serialization failure and cannot be masked by the duplicate gate.
        /// </summary>
        static void ApplySchemaMutation(List<ExportMember> members)
        {
            if (SchemaMutation == Common2DMemberSchemaMutation.None) return;
            if (members == null || members.Count == 0) return;
            int index = SchemaMutationMemberIndex;
            if (index < 0 || index >= members.Count) return;
            ExportMember member = members[index];
            switch (SchemaMutation)
            {
                case Common2DMemberSchemaMutation.Name:
                    member.Name = member.Name + ".mutated";
                    break;
                case Common2DMemberSchemaMutation.Order:
                    int other = index + 1 < members.Count ? index + 1 : index - 1;
                    members[index] = members[other];
                    members[other] = member;
                    break;
                case Common2DMemberSchemaMutation.Group:
                    member.Group = member.Group + "_mutated";
                    break;
                case Common2DMemberSchemaMutation.Dtype:
                    member.Dtype = member.Dtype == "<f8" ? "<f4" : "<f8";
                    break;
                case Common2DMemberSchemaMutation.Shape:
                    member.Shape = "[1]";
                    break;
                case Common2DMemberSchemaMutation.ElementCount:
                    member.ElementCount = member.ElementCount + 1;
                    break;
                case Common2DMemberSchemaMutation.ByteCount:
                    var shortened = new byte[Math.Max(0, member.Bytes.Length - 1)];
                    Array.Copy(member.Bytes, shortened, shortened.Length);
                    member.Bytes = shortened;
                    break;
                case Common2DMemberSchemaMutation.PadBit:
                    if (member.Bytes.Length > 0)
                    {
                        var padded = (byte[])member.Bytes.Clone();
                        padded[padded.Length - 1] |= 0x80;
                        member.Bytes = padded;
                    }
                    break;
            }
        }

        /// <summary>
        /// Deterministic lowercase 64-hex identity over the fixed prefix and the capture
        /// binding only. It carries no path and no user information.
        /// </summary>
        public static string ComputeExportId(
            string captureSessionId, string requestedTarget,
            string artifactVersion, string artifactSha256)
        {
            string material = ExportIdPrefix + "\n" +
                (captureSessionId ?? string.Empty) + "\n" +
                (requestedTarget ?? string.Empty) + "\n" +
                (artifactVersion ?? string.Empty) + "\n" +
                (artifactSha256 ?? string.Empty) + "\n";
            return Sha256Hex(new UTF8Encoding(false).GetBytes(material));
        }

        /// <summary>
        /// The production developer route. It always writes to the fixed validation root
        /// and accepts no caller-supplied location.
        /// </summary>
        public static Common2DDiagnosticExportResult ExportToDeveloperRoot(
            Common2DDiagnosticTrace trace)
        {
            return ExportCore(trace, DeveloperExportRoot, true);
        }

        /// <summary>
        /// The explicit, documented test seam. It accepts an isolated scratch root and
        /// still fail-closes on any root that resolves inside a project or persistence
        /// scope, so a test cannot write into a protected location either.
        /// </summary>
        public static Common2DDiagnosticExportResult ExportToScratchRoot(
            Common2DDiagnosticTrace trace, string scratchRoot)
        {
            return ExportCore(trace, scratchRoot, false);
        }

        static Common2DDiagnosticExportResult ExportCore(
            Common2DDiagnosticTrace trace, string exportRoot, bool developerRoute)
        {
            var result = new Common2DDiagnosticExportResult();
            if (trace == null)
            {
                result.Status = Common2DTraceStatus.Failed;
                result.FailureReason = Common2DTraceFailureReason.NotArmed;
                result.Marker = Common2DDiagnosticLogMarker.BuildFailure(
                    null, result.Status, result.FailureReason);
                return result;
            }

            result.ExportId = ComputeExportId(
                trace.CaptureSessionId, trace.RequestedTarget,
                trace.ArtifactVersion, trace.ArtifactSha256);

            // Full semantic validation of every binding, shape, count and the accepted
            // terminal seal happens before any path is touched or any file is created.
            Common2DTraceFailureReason validation;
            if (!trace.Validate(out validation))
            {
                return Finish(result, trace, Common2DTraceStatus.Failed, validation);
            }
            string resolvedRoot;
            if (!TryResolveRoot(exportRoot, developerRoute, out resolvedRoot))
            {
                return Finish(result, trace, Common2DTraceStatus.Failed,
                              Common2DTraceFailureReason.ExportRootRejected);
            }
            if (Common2DDiagnosticTraceBuffer.HasCommitted(trace.CaptureSessionId))
            {
                return Finish(result, trace, Common2DTraceStatus.Duplicate,
                              Common2DTraceFailureReason.AlreadyCommitted);
            }

            List<ExportMember> members;
            byte[] manifestBytes;
            try
            {
                if (FaultInjection == Common2DExportFaultInjection.SerializationThrows)
                    throw new InvalidOperationException("injected serialization fault");
                members = BuildMembers(trace);
                manifestBytes = BuildManifestBytes(trace, result.ExportId, members);
            }
            catch (Exception)
            {
                return Finish(result, trace, Common2DTraceStatus.Failed,
                              Common2DTraceFailureReason.SerializationFailed);
            }

            ApplySchemaMutation(members);

            // Exact schema validation happens BEFORE any root or staging path is
            // created, and before the duplicate-session gate can mask it.
            if (!ValidateCanonicalSchema(members, trace) || !ValidateMembers(members))
            {
                return Finish(result, trace, Common2DTraceStatus.Failed,
                              Common2DTraceFailureReason.InvalidPayloadShape);
            }

            result.PayloadSha256 = Sha256Hex(manifestBytes);
            result.ArrayShapes = DescribeShapes(members);
            result.ArraySha256s = DescribeDigests(members);
            result.MemberCount = members.Count;

            string root = resolvedRoot;
            string finalDirectory = Path.Combine(root, result.ExportId);
            string staging = Path.Combine(root, TemporaryPrefix + result.ExportId);

            if (Directory.Exists(finalDirectory))
            {
                return Finish(result, trace, Common2DTraceStatus.Duplicate,
                              Common2DTraceFailureReason.AlreadyCommitted);
            }
            if (Directory.Exists(staging))
            {
                // A pre-existing staging directory belongs to another invocation. It is
                // never merged into, truncated or removed here.
                return Finish(result, trace, Common2DTraceStatus.Failed,
                              Common2DTraceFailureReason.StagingConflict);
            }

            bool stagingOwnedByThisCall = false;
            try
            {
                if (FaultInjection == Common2DExportFaultInjection.WriteThrows)
                    throw new IOException("injected write fault");

                Directory.CreateDirectory(root);
                Directory.CreateDirectory(staging);
                stagingOwnedByThisCall = true;

                for (int i = 0; i < members.Count; i++)
                {
                    string target = Path.Combine(staging, members[i].Name);
                    using (var stream = new FileStream(
                        target, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                    {
                        stream.Write(members[i].Bytes, 0, members[i].Bytes.Length);
                    }
                }
                using (var stream = new FileStream(
                    Path.Combine(staging, ManifestName),
                    FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    stream.Write(manifestBytes, 0, manifestBytes.Length);
                }

                if (!ValidateStaging(staging, members, manifestBytes))
                {
                    TryDeleteStaging(staging, stagingOwnedByThisCall);
                    return Finish(result, trace, Common2DTraceStatus.Failed,
                                  Common2DTraceFailureReason.IncompletePayload);
                }
                if (Directory.Exists(finalDirectory))
                {
                    TryDeleteStaging(staging, stagingOwnedByThisCall);
                    return Finish(result, trace, Common2DTraceStatus.Duplicate,
                                  Common2DTraceFailureReason.AlreadyCommitted);
                }
                Directory.Move(staging, finalDirectory);
                stagingOwnedByThisCall = false;
            }
            catch (Exception)
            {
                TryDeleteStaging(staging, stagingOwnedByThisCall);
                return Finish(result, trace, Common2DTraceStatus.Failed,
                              Common2DTraceFailureReason.WriteFailed);
            }

            Common2DDiagnosticTraceBuffer.MarkCommitted(trace.CaptureSessionId);
            return Finish(result, trace, Common2DTraceStatus.Committed,
                          Common2DTraceFailureReason.None);
        }

        static Common2DDiagnosticExportResult Finish(
            Common2DDiagnosticExportResult result, Common2DDiagnosticTrace trace,
            Common2DTraceStatus status, Common2DTraceFailureReason reason)
        {
            result.Status = status;
            result.FailureReason = reason;
            if (status == Common2DTraceStatus.Committed)
            {
                string marker;
                Common2DTraceFailureReason markerReason;
                if (Common2DDiagnosticLogMarker.TryBuildSuccess(
                        trace, result.ExportId, result.PayloadSha256,
                        result.ArrayShapes, result.ArraySha256s,
                        out marker, out markerReason))
                {
                    result.Marker = marker;
                    return result;
                }
                // A marker that cannot be emitted completely is reported as a typed
                // failure; it is never shortened into something that looks successful.
                result.Status = Common2DTraceStatus.Failed;
                result.FailureReason = markerReason;
                result.Marker = Common2DDiagnosticLogMarker.BuildFailure(
                    trace, result.Status, markerReason);
                return result;
            }
            result.Marker = Common2DDiagnosticLogMarker.BuildFailure(
                trace, status, reason);
            return result;
        }

        /// <summary>
        /// Resolves and fail-closes the export root. The developer route accepts only the
        /// one fixed validation root; any route rejects a root that escapes upward,
        /// resolves through a symlink or reparse point, or lands inside a project or
        /// protected persistence scope.
        /// </summary>
        static bool TryResolveRoot(
            string exportRoot, bool developerRoute, out string resolved)
        {
            resolved = null;
            try
            {
                if (developerRoute) exportRoot = DeveloperExportRoot;
                if (string.IsNullOrEmpty(exportRoot)) return false;
                if (exportRoot.IndexOf("..", StringComparison.Ordinal) >= 0) return false;

                string full = Path.GetFullPath(exportRoot);
                if (developerRoute)
                {
                    string expected = Path.GetFullPath(DeveloperExportRoot);
                    if (!string.Equals(full, expected, StringComparison.OrdinalIgnoreCase))
                        return false;
                }
                else if (!Path.IsPathRooted(exportRoot))
                {
                    // A scratch root must be an absolute, non-project location.
                    return false;
                }

                string[] segments = full.Split(
                    Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                for (int i = 0; i < segments.Length; i++)
                {
                    for (int r = 0; r < ProtectedRootNames.Length; r++)
                    {
                        if (string.Equals(segments[i], ProtectedRootNames[r],
                                          StringComparison.OrdinalIgnoreCase))
                            return false;
                    }
                }
                if (HasReparsePointOnPath(full)) return false;
                resolved = full;
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>Any symlink or junction on the resolved chain is refused.</summary>
        static bool HasReparsePointOnPath(string full)
        {
            var directory = new DirectoryInfo(full);
            while (directory != null)
            {
                if (directory.Exists &&
                    (directory.Attributes & FileAttributes.ReparsePoint) != 0)
                    return true;
                directory = directory.Parent;
            }
            return false;
        }

        /// <summary>
        /// Every declared shape product, element count and byte count must agree, every
        /// packed bool block must have zero padding bits, member names must be unique and
        /// all ten canonical groups must be represented, before anything is written.
        /// </summary>
        /// <summary>
        /// Exactly twenty-four members in canonical order, each with its exact name,
        /// group, dtype and resolved shape. An extra, missing, reordered or renamed
        /// member is refused before anything is written.
        /// </summary>
        static bool ValidateCanonicalSchema(
            List<ExportMember> members, Common2DDiagnosticTrace trace)
        {
            if (members == null || members.Count != CanonicalMemberCount) return false;
            int rawCount = trace.RawObservations.Count;
            int normalizedFrames = trace.NormalizedFrameCount;
            for (int i = 0; i < CanonicalMemberCount; i++)
            {
                string[] expected = CanonicalMemberSchema[i];
                ExportMember member = members[i];
                if (!string.Equals(member.Name, expected[0], StringComparison.Ordinal))
                    return false;
                if (!string.Equals(member.Group, expected[1], StringComparison.Ordinal))
                    return false;
                if (!string.Equals(member.Dtype, expected[2], StringComparison.Ordinal))
                    return false;
                string shape = expected[3]
                    .Replace("N", rawCount.ToString(CultureInfo.InvariantCulture))
                    .Replace("T", normalizedFrames.ToString(CultureInfo.InvariantCulture));
                if (!string.Equals(member.Shape, shape, StringComparison.Ordinal))
                    return false;
            }
            return true;
        }

        static bool ValidateMembers(List<ExportMember> members)
        {
            var seen = new HashSet<string>();
            for (int i = 0; i < members.Count; i++)
            {
                ExportMember member = members[i];
                if (!seen.Add(member.Name)) return false;
                if (member.Dtype == "utf8_json")
                {
                    if (member.ElementCount != 1 || member.Bytes.Length <= 0) return false;
                    continue;
                }
                long product = ShapeProduct(member.Shape);
                if (product < 0 || product != member.ElementCount) return false;
                if (member.Dtype == "<f8")
                {
                    if (member.Bytes.Length != member.ElementCount * 8) return false;
                }
                else if (member.Dtype == "packed_bool_lsb_first")
                {
                    long expected = (member.ElementCount + 7) / 8;
                    if (member.Bytes.Length != expected) return false;
                    int padding = (int)(expected * 8 - member.ElementCount);
                    if (padding > 0 && member.Bytes.Length > 0)
                    {
                        int last = member.Bytes[member.Bytes.Length - 1];
                        for (int bit = 8 - padding; bit < 8; bit++)
                            if ((last & (1 << bit)) != 0) return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            for (int g = 0; g < CanonicalPayloadGroups.Length; g++)
            {
                bool present = false;
                for (int i = 0; i < members.Count && !present; i++)
                    present = members[i].Group == CanonicalPayloadGroups[g];
                if (!present) return false;
            }
            return true;
        }

        static long ShapeProduct(string shape)
        {
            if (string.IsNullOrEmpty(shape) || shape[0] != '[' ||
                shape[shape.Length - 1] != ']') return -1;
            string body = shape.Substring(1, shape.Length - 2);
            if (body.Length == 0) return 1;
            string[] parts = body.Split(',');
            long product = 1;
            for (int i = 0; i < parts.Length; i++)
            {
                long value;
                if (!long.TryParse(parts[i], NumberStyles.None,
                                   CultureInfo.InvariantCulture, out value)) return -1;
                product *= value;
            }
            return product;
        }

        /// <summary>Removes only a staging directory this invocation created.</summary>
        static void TryDeleteStaging(string staging, bool ownedByThisCall)
        {
            if (!ownedByThisCall) return;
            try
            {
                if (Directory.Exists(staging)) Directory.Delete(staging, true);
            }
            catch (Exception)
            {
                // A staging directory that cannot be removed is left alone. It is never
                // merged into and never becomes a published export.
            }
        }

        static bool ValidateStaging(
            string staging, List<ExportMember> members, byte[] manifestBytes)
        {
            for (int i = 0; i < members.Count; i++)
            {
                string target = Path.Combine(staging, members[i].Name);
                if (!File.Exists(target)) return false;
                byte[] written = File.ReadAllBytes(target);
                if (written.Length != members[i].Bytes.Length) return false;
                if (!string.Equals(Sha256Hex(written), members[i].Sha256, StringComparison.Ordinal))
                    return false;
            }
            string manifestPath = Path.Combine(staging, ManifestName);
            if (!File.Exists(manifestPath)) return false;
            byte[] storedManifest = File.ReadAllBytes(manifestPath);
            if (storedManifest.Length != manifestBytes.Length) return false;
            for (int i = 0; i < storedManifest.Length; i++)
                if (storedManifest[i] != manifestBytes[i]) return false;
            return true;
        }

        // ------------------------------------------------------------------ members
        static List<ExportMember> BuildMembers(Common2DDiagnosticTrace trace)
        {
            int observationCount = trace.RawObservations.Count;
            var timestamps = new double[observationCount];
            var pose = new double[observationCount * 33 * 3];
            var left = new double[observationCount * 21 * 3];
            var right = new double[observationCount * 21 * 3];
            var poseValid = new bool[observationCount * 33];
            var leftValid = new bool[observationCount * 21];
            var rightValid = new bool[observationCount * 21];
            var poseReappear = new bool[observationCount * 33];
            var leftReappear = new bool[observationCount * 21];
            var rightReappear = new bool[observationCount * 21];
            var poseBoundary = new bool[observationCount * 33];
            var leftBoundary = new bool[observationCount * 21];
            var rightBoundary = new bool[observationCount * 21];

            for (int i = 0; i < observationCount; i++)
            {
                Common2DRawObservationRecord record = trace.RawObservations[i];
                timestamps[i] = record.TimestampSeconds;
                Array.Copy(record.PoseCoordinates, 0, pose, i * 33 * 3, 33 * 3);
                Array.Copy(record.LeftHandCoordinates, 0, left, i * 21 * 3, 21 * 3);
                Array.Copy(record.RightHandCoordinates, 0, right, i * 21 * 3, 21 * 3);
                Array.Copy(record.PoseValidMask, 0, poseValid, i * 33, 33);
                Array.Copy(record.LeftHandValidMask, 0, leftValid, i * 21, 21);
                Array.Copy(record.RightHandValidMask, 0, rightValid, i * 21, 21);
                Array.Copy(record.PoseReappearanceMask, 0, poseReappear, i * 33, 33);
                Array.Copy(record.LeftHandReappearanceMask, 0, leftReappear, i * 21, 21);
                Array.Copy(record.RightHandReappearanceMask, 0, rightReappear, i * 21, 21);
                Array.Copy(record.PoseStrongDepthBoundaryMask, 0, poseBoundary, i * 33, 33);
                Array.Copy(record.LeftHandStrongDepthBoundaryMask, 0, leftBoundary, i * 21, 21);
                Array.Copy(record.RightHandStrongDepthBoundaryMask, 0, rightBoundary, i * 21, 21);
            }

            const string raw = "raw_observation_sequence";
            var members = new List<ExportMember>();
            members.Add(Float64("raw_timestamps.bin", "[" + observationCount + "]", timestamps, raw));
            members.Add(Float64("raw_pose_coordinates.bin", "[" + observationCount + ",33,3]", pose, raw));
            members.Add(Float64("raw_left_hand_coordinates.bin", "[" + observationCount + ",21,3]", left, raw));
            members.Add(Float64("raw_right_hand_coordinates.bin", "[" + observationCount + ",21,3]", right, raw));
            members.Add(Bools("raw_pose_valid_mask.bin", "[" + observationCount + ",33]", poseValid, raw));
            members.Add(Bools("raw_left_hand_valid_mask.bin", "[" + observationCount + ",21]", leftValid, raw));
            members.Add(Bools("raw_right_hand_valid_mask.bin", "[" + observationCount + ",21]", rightValid, raw));
            members.Add(Bools("raw_pose_reappearance_mask.bin", "[" + observationCount + ",33]", poseReappear, raw));
            members.Add(Bools("raw_left_hand_reappearance_mask.bin", "[" + observationCount + ",21]", leftReappear, raw));
            members.Add(Bools("raw_right_hand_reappearance_mask.bin", "[" + observationCount + ",21]", rightReappear, raw));
            members.Add(Bools("raw_pose_strong_depth_boundary_mask.bin", "[" + observationCount + ",33]", poseBoundary, raw));
            members.Add(Bools("raw_left_hand_strong_depth_boundary_mask.bin", "[" + observationCount + ",21]", leftBoundary, raw));
            members.Add(Bools("raw_right_hand_strong_depth_boundary_mask.bin", "[" + observationCount + ",21]", rightBoundary, raw));
            members.Add(Json("raw_convention.json", ConventionJson(trace), raw));
            members.Add(Json("raw_geometry.json", GeometryJson(trace), raw));

            int normalizedFrames = trace.NormalizedFrameCount;
            members.Add(Float64("normalized_coordinates.bin",
                "[" + normalizedFrames + ",16,3]", trace.NormalizedCoordinates,
                "normalized_common2d_coordinates"));
            members.Add(Bools("input_mask.bin", "[" + normalizedFrames + ",16]",
                trace.InputMask, "common2d_input_mask"));
            members.Add(Float64("resampled_coordinates.bin", "[32,16,2]",
                trace.ResampledCoordinates, "resampled_common2d_coordinates"));
            members.Add(Bools("resampled_mask.bin", "[32,16]",
                trace.ResampledMask, "resampled_common2d_mask"));
            members.Add(Float64("feature_vector.bin", "[1536]",
                trace.FeatureVector, "feature_vector"));
            members.Add(Float64("standardized_feature_vector.bin", "[1536]",
                trace.StandardizedFeatureVector, "standardized_feature_vector"));
            members.Add(Float64("pca_projection.bin", "[4]",
                trace.PcaProjection, "pca_projection"));
            members.Add(Float64("lr_component_contributions.bin", "[4]",
                trace.LrComponentContributions, "lr_component_contributions"));
            members.Add(Float64("lr_scalars.bin", "[4]", new[]
            {
                trace.LrIntercept, trace.DecisionValue,
                trace.GreetingProbability, trace.ThankYouProbability,
            }, "lr_intercept_decision_probabilities"));
            return members;
        }

        static ExportMember Float64(string name, string shape, double[] values, string group)
        {
            var bytes = new byte[values.Length * 8];
            for (int i = 0; i < values.Length; i++)
            {
                double value = values[i];
                ulong bits = double.IsNaN(value)
                    ? CanonicalQuietNaNBits
                    : unchecked((ulong)BitConverter.DoubleToInt64Bits(value));
                int offset = i * 8;
                for (int b = 0; b < 8; b++) bytes[offset + b] = (byte)((bits >> (8 * b)) & 0xFF);
            }
            return new ExportMember
            {
                Name = name, Dtype = "<f8", Shape = shape, ElementCount = values.Length,
                Bytes = bytes, Sha256 = Sha256Hex(bytes), Group = group,
            };
        }

        static ExportMember Bools(string name, string shape, bool[] values, string group)
        {
            var bytes = new byte[(values.Length + 7) / 8];
            for (int i = 0; i < values.Length; i++)
                if (values[i]) bytes[i / 8] |= (byte)(1 << (i % 8));
            return new ExportMember
            {
                Name = name, Dtype = "packed_bool_lsb_first", Shape = shape,
                ElementCount = values.Length, Bytes = bytes, Sha256 = Sha256Hex(bytes),
                Group = group,
            };
        }

        static ExportMember Json(string name, string text, string group)
        {
            byte[] bytes = new UTF8Encoding(false).GetBytes(text);
            return new ExportMember
            {
                Name = name, Dtype = "utf8_json", Shape = "[]", ElementCount = 1,
                Bytes = bytes, Sha256 = Sha256Hex(bytes), Group = group,
            };
        }

        // ------------------------------------------------------------------ json
        static string ConventionJson(Common2DDiagnosticTrace trace)
        {
            var text = new StringBuilder(512);
            text.Append("{\n \"observations\": [\n");
            for (int i = 0; i < trace.RawObservations.Count; i++)
            {
                Common2DConventionRecord convention = trace.RawObservations[i].Convention;
                if (i > 0) text.Append(",\n");
                text.Append("  {");
                text.Append("\"id\": ").Append(Quote(convention.Id));
                text.Append(", \"coordinateSpace\": ").Append(Quote(convention.CoordinateSpace));
                text.Append(", \"axisAndDepth\": ").Append(Quote(convention.AxisAndDepth));
                text.Append(", \"handedness\": ").Append(Quote(convention.Handedness));
                text.Append(", \"mirror\": ").Append(Quote(convention.Mirror));
                text.Append("}");
            }
            text.Append("\n ]\n}\n");
            return text.ToString();
        }

        static string GeometryJson(Common2DDiagnosticTrace trace)
        {
            var text = new StringBuilder(4096);
            text.Append("{\n \"observations\": [\n");
            for (int i = 0; i < trace.RawObservations.Count; i++)
            {
                Common2DRawObservationRecord record = trace.RawObservations[i];
                if (i > 0) text.Append(",\n");
                text.Append("  {\n   \"hand\": ");
                AppendGeometry(text, record.MergedGeometryHand);
                text.Append(",\n   \"pose\": ");
                AppendGeometry(text, record.MergedGeometryPose);
                text.Append(",\n   \"agreement\": ");
                AppendAgreement(text, record.MergedGeometryAgreement);
                text.Append("\n  }");
            }
            text.Append("\n ]\n}\n");
            return text.ToString();
        }

        static void AppendGeometry(StringBuilder text, Common2DGeometryRecord geometry)
        {
            text.Append("{");
            text.Append("\"present\": ").Append(geometry.Present ? "true" : "false");
            text.Append(", \"provenance\": ").Append(Quote(geometry.Provenance));
            text.Append(", \"observedWebcamWidth\": ").Append(Integer(geometry.ObservedWebcamWidth));
            text.Append(", \"observedWebcamHeight\": ").Append(Integer(geometry.ObservedWebcamHeight));
            text.Append(", \"observedVideoRotationAngleOrNull\": ").Append(Quote(geometry.ObservedVideoRotationAngleOrNull));
            text.Append(", \"observedVideoVerticallyMirroredOrNull\": ").Append(Quote(geometry.ObservedVideoVerticallyMirroredOrNull));
            text.Append(", \"inferenceTextureWidth\": ").Append(Integer(geometry.InferenceTextureWidth));
            text.Append(", \"inferenceTextureHeight\": ").Append(Integer(geometry.InferenceTextureHeight));
            text.Append(", \"inferenceImageWidth\": ").Append(Integer(geometry.InferenceImageWidth));
            text.Append(", \"inferenceImageHeight\": ").Append(Integer(geometry.InferenceImageHeight));
            text.Append(", \"imageFormat\": ").Append(Quote(geometry.ImageFormat));
            text.Append(", \"preInferenceHorizontalFlip\": ").Append(Quote(geometry.PreInferenceHorizontalFlip));
            text.Append(", \"preInferenceVerticalFlip\": ").Append(Quote(geometry.PreInferenceVerticalFlip));
            text.Append(", \"preInferenceRotationDegreesOrNull\": ").Append(Quote(geometry.PreInferenceRotationDegreesOrNull));
            text.Append(", \"crop\": ").Append(Quote(geometry.Crop));
            text.Append(", \"resize\": ").Append(Quote(geometry.Resize));
            text.Append(", \"letterbox\": ").Append(Quote(geometry.Letterbox));
            text.Append(", \"pixelBufferRowOrder\": ").Append(Quote(geometry.PixelBufferRowOrder));
            text.Append(", \"providerFrameIndexOrNull\": ").Append(Quote(geometry.ProviderFrameIndexOrNull));
            text.Append(", \"mediaPipeTimestampMs\": ").Append(
                geometry.MediaPipeTimestampMs.ToString(CultureInfo.InvariantCulture));
            text.Append(", \"unityFrameCountOrNull\": ").Append(Quote(geometry.UnityFrameCountOrNull));
            text.Append("}");
        }

        static void AppendAgreement(StringBuilder text, Common2DGeometryAgreementRecord agreement)
        {
            text.Append("{");
            text.Append("\"present\": ").Append(agreement.Present ? "true" : "false");
            text.Append(", \"bothAvailable\": ").Append(agreement.BothAvailable ? "true" : "false");
            text.Append(", \"fullContractCompatible\": ").Append(agreement.FullContractCompatible ? "true" : "false");
            text.Append(", \"firstMismatchDescription\": ").Append(Quote(agreement.FirstMismatchDescription));
            text.Append(", \"handPoseTimestampSkewMs\": ").Append(Quote(agreement.HandPoseTimestampSkewMs));
            text.Append(", \"handResultStatus\": ").Append(Quote(agreement.HandResultStatus));
            text.Append(", \"poseResultStatus\": ").Append(Quote(agreement.PoseResultStatus));
            text.Append(", \"handResultDiagnosticCode\": ").Append(Quote(agreement.HandResultDiagnosticCode));
            text.Append(", \"poseResultDiagnosticCode\": ").Append(Quote(agreement.PoseResultDiagnosticCode));
            text.Append("}");
        }

        /// <summary>
        /// The canonical manifest. Fixed key order, LF endings, UTF-8 without BOM,
        /// invariant-culture scalars. It excludes its own digest and lists every other
        /// member exactly once.
        /// </summary>
        static byte[] BuildManifestBytes(
            Common2DDiagnosticTrace trace, string exportId, List<ExportMember> members)
        {
            var text = new StringBuilder(8192);
            text.Append("{\n");
            text.Append(" \"schema\": \"words_in_silence.common2d.diagnostic_trace.v1_0f2\",\n");
            text.Append(" \"step\": \"10P_v1.0f2\",\n");
            text.Append(" \"exportId\": ").Append(Quote(exportId)).Append(",\n");
            text.Append(" \"binding\": {\n");
            text.Append("  \"captureSessionId\": ").Append(Quote(trace.CaptureSessionId)).Append(",\n");
            text.Append("  \"requestedTarget\": ").Append(Quote(trace.RequestedTarget)).Append(",\n");
            text.Append("  \"artifactVersion\": ").Append(Quote(trace.ArtifactVersion)).Append(",\n");
            text.Append("  \"artifactSha256\": ").Append(Quote(trace.ArtifactSha256)).Append(",\n");
            text.Append("  \"sourceFrameCount\": ").Append(Integer(trace.SourceFrameCount)).Append(",\n");
            text.Append("  \"domainContract\": ").Append(Quote(trace.DomainContract)).Append("\n");
            text.Append(" },\n");
            text.Append(" \"scalars\": {\n");
            text.Append("  \"lrIntercept\": ").Append(Number(trace.LrIntercept)).Append(",\n");
            text.Append("  \"decisionValue\": ").Append(Number(trace.DecisionValue)).Append(",\n");
            text.Append("  \"greetingProbability\": ").Append(Number(trace.GreetingProbability)).Append(",\n");
            text.Append("  \"thankYouProbability\": ").Append(Number(trace.ThankYouProbability)).Append("\n");
            text.Append(" },\n");
            text.Append(" \"nanRepresentation\": {\n");
            text.Append("  \"jsonToken\": ").Append(Quote(JsonNaNSentinel)).Append(",\n");
            text.Append("  \"binaryBitPattern\": \"0x7ff8000000000000\",\n");
            text.Append("  \"coercedToZero\": false\n");
            text.Append(" },\n");
            text.Append(" \"canonicalPayloadGroups\": [\n");
            for (int i = 0; i < CanonicalPayloadGroups.Length; i++)
            {
                text.Append("  ").Append(Quote(CanonicalPayloadGroups[i]));
                text.Append(i == CanonicalPayloadGroups.Length - 1 ? "\n" : ",\n");
            }
            text.Append(" ],\n");
            text.Append(" \"rawObservationFieldCount\": ")
                .Append(Integer(Common2DRawObservationRecord.FieldCount)).Append(",\n");
            text.Append(" \"observationCount\": ")
                .Append(Integer(trace.RawObservations.Count)).Append(",\n");
            text.Append(" \"manifestExcludesItsOwnDigest\": true,\n");
            text.Append(" \"memberCount\": ").Append(Integer(members.Count)).Append(",\n");
            text.Append(" \"members\": [\n");
            for (int i = 0; i < members.Count; i++)
            {
                ExportMember member = members[i];
                text.Append("  {");
                text.Append("\"name\": ").Append(Quote(member.Name));
                text.Append(", \"group\": ").Append(Quote(member.Group));
                text.Append(", \"dtype\": ").Append(Quote(member.Dtype));
                text.Append(", \"shape\": ").Append(Quote(member.Shape));
                text.Append(", \"elementCount\": ").Append(member.ElementCount.ToString(CultureInfo.InvariantCulture));
                text.Append(", \"byteCount\": ").Append(member.Bytes.Length.ToString(CultureInfo.InvariantCulture));
                text.Append(", \"sha256\": ").Append(Quote(member.Sha256));
                text.Append("}");
                text.Append(i == members.Count - 1 ? "\n" : ",\n");
            }
            text.Append(" ]\n");
            text.Append("}\n");
            return new UTF8Encoding(false).GetBytes(text.ToString());
        }

        /// <summary>Every member shape, complete, in the fixed member order.</summary>
        static string DescribeShapes(List<ExportMember> members)
        {
            var text = new StringBuilder(1024);
            for (int i = 0; i < members.Count; i++)
            {
                if (i > 0) text.Append('|');
                text.Append(members[i].Dtype == "utf8_json" ? "[1]" : members[i].Shape);
            }
            return text.ToString();
        }

        /// <summary>Every member digest, complete and unabridged, in the same order.</summary>
        static string DescribeDigests(List<ExportMember> members)
        {
            var text = new StringBuilder(2048);
            for (int i = 0; i < members.Count; i++)
            {
                if (i > 0) text.Append('|');
                text.Append(members[i].Sha256);
            }
            return text.ToString();
        }

        // ------------------------------------------------------------------ scalars
        /// <summary>
        /// Round-trip-exact JSON number. A non-finite scalar is written as the exact
        /// string sentinel; a bare NaN or Infinity literal is never emitted.
        /// </summary>
        static string Number(double value)
        {
            if (double.IsNaN(value)) return Quote(JsonNaNSentinel);
            if (double.IsPositiveInfinity(value)) return Quote("Infinity");
            if (double.IsNegativeInfinity(value)) return Quote("-Infinity");
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        static string Integer(long value)
        {
            return value.ToString(CultureInfo.InvariantCulture);
        }

        static string Quote(string value)
        {
            var text = new StringBuilder((value == null ? 0 : value.Length) + 2);
            text.Append('"');
            if (value != null)
            {
                for (int i = 0; i < value.Length; i++)
                {
                    char character = value[i];
                    switch (character)
                    {
                        case '"': text.Append("\\\""); break;
                        case '\\': text.Append("\\\\"); break;
                        case '\n': text.Append("\\n"); break;
                        case '\r': text.Append("\\r"); break;
                        case '\t': text.Append("\\t"); break;
                        default:
                            if (character < ' ')
                                text.Append("\\u").Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                            else text.Append(character);
                            break;
                    }
                }
            }
            text.Append('"');
            return text.ToString();
        }

        static string Sha256Hex(byte[] bytes)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(bytes);
                var text = new StringBuilder(digest.Length * 2);
                for (int i = 0; i < digest.Length; i++)
                    text.Append(digest[i].ToString("x2", CultureInfo.InvariantCulture));
                return text.ToString();
            }
        }
    }
}
#endif
