using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    public sealed class RecoveredPhysicalCapture
    {
        public string EvidencePath { get; }
        public string ChecksumPath { get; }
        public string CaptureId { get; }
        public string ExpectedWord { get; }
        public int TakeNumber { get; }
        public string CaptureTimeUtc { get; }
        public bool ObjectivelyValid { get; }

        internal RecoveredPhysicalCapture(
            string evidencePath,
            string checksumPath,
            string captureId,
            string expectedWord,
            int takeNumber,
            string captureTimeUtc,
            bool objectivelyValid)
        {
            EvidencePath = evidencePath;
            ChecksumPath = checksumPath;
            CaptureId = captureId;
            ExpectedWord = expectedWord;
            TakeNumber = takeNumber;
            CaptureTimeUtc = captureTimeUtc;
            ObjectivelyValid = objectivelyValid;
        }
    }

    public sealed class PhysicalEvidenceResumeReport
    {
        readonly ReadOnlyCollection<RecoveredPhysicalCapture> verifiedCaptures;
        readonly ReadOnlyCollection<RecoveredPhysicalCapture> recoveredValidCaptures;

        public string EvidenceDirectory { get; }
        public int JsonFileCount { get; }
        public int ChecksumFileCount { get; }
        public IReadOnlyList<RecoveredPhysicalCapture> VerifiedCaptures => verifiedCaptures;
        public IReadOnlyList<RecoveredPhysicalCapture> RecoveredValidCaptures => recoveredValidCaptures;

        internal PhysicalEvidenceResumeReport(
            string evidenceDirectory,
            int jsonFileCount,
            int checksumFileCount,
            IList<RecoveredPhysicalCapture> verifiedCaptures,
            IList<RecoveredPhysicalCapture> recoveredValidCaptures)
        {
            EvidenceDirectory = evidenceDirectory;
            JsonFileCount = jsonFileCount;
            ChecksumFileCount = checksumFileCount;
            this.verifiedCaptures = new ReadOnlyCollection<RecoveredPhysicalCapture>(verifiedCaptures.ToArray());
            this.recoveredValidCaptures = new ReadOnlyCollection<RecoveredPhysicalCapture>(recoveredValidCaptures.ToArray());
        }
    }

    /// <summary>Validates existing evidence without opening any file for write and restores only the first valid fixed take.</summary>
    public static class PhysicalCaptureEvidenceResume
    {
        static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(false, true);

        public static PhysicalEvidenceResumeReport VerifyReadOnly(string evidenceDirectory)
        {
            if (string.IsNullOrWhiteSpace(evidenceDirectory)) throw new ArgumentException("evidence directory is required", nameof(evidenceDirectory));
            string fullDirectory = Path.GetFullPath(evidenceDirectory);
            if (!Directory.Exists(fullDirectory))
                return new PhysicalEvidenceResumeReport(fullDirectory, 0, 0, new List<RecoveredPhysicalCapture>(), new List<RecoveredPhysicalCapture>());

            string[] jsonPaths = Directory.GetFiles(fullDirectory, "*.json", SearchOption.TopDirectoryOnly)
                .OrderBy(path => Path.GetFileName(path), StringComparer.Ordinal).ToArray();
            string[] checksumPaths = Directory.GetFiles(fullDirectory, "*.json.sha256", SearchOption.TopDirectoryOnly)
                .OrderBy(path => Path.GetFileName(path), StringComparer.Ordinal).ToArray();
            var expectedChecksums = new HashSet<string>(jsonPaths.Select(path => Path.GetFullPath(path + ".sha256")), StringComparer.OrdinalIgnoreCase);
            var actualChecksums = new HashSet<string>(checksumPaths.Select(Path.GetFullPath), StringComparer.OrdinalIgnoreCase);
            if (!expectedChecksums.SetEquals(actualChecksums))
                throw new InvalidDataException($"evidence JSON/checksum pair set differs: json={jsonPaths.Length}, checksum={checksumPaths.Length}");

            var verified = new List<RecoveredPhysicalCapture>(jsonPaths.Length);
            foreach (string jsonPath in jsonPaths)
            {
                string checksumPath = jsonPath + ".sha256";
                if (!PhysicalCaptureEvidenceFile.Verify(jsonPath, checksumPath))
                    throw new InvalidDataException("evidence SHA-256 verification failed: " + Path.GetFileName(jsonPath));
                string json = File.ReadAllText(jsonPath, Utf8NoBom);
                ResumeEvidenceDto dto;
                try { dto = JsonUtility.FromJson<ResumeEvidenceDto>(json); }
                catch (Exception exception) { throw new InvalidDataException("evidence JSON parse failed: " + Path.GetFileName(jsonPath), exception); }
                ValidateSchema(dto, jsonPath);
                verified.Add(new RecoveredPhysicalCapture(
                    Path.GetFullPath(jsonPath),
                    Path.GetFullPath(checksumPath),
                    dto.capture_id,
                    dto.expected_word,
                    dto.take_number,
                    dto.capture_time_utc,
                    dto.pre_export_objectively_valid));
            }

            var recovered = new List<RecoveredPhysicalCapture>(Step9CPhysicalCompatibilityConstants.FixedWords.Length);
            for (int takeIndex = 0; takeIndex < Step9CPhysicalCompatibilityConstants.FixedWords.Length; takeIndex++)
            {
                int takeNumber = takeIndex + 1;
                string expectedWord = Step9CPhysicalCompatibilityConstants.FixedWords[takeIndex];
                RecoveredPhysicalCapture firstValid = verified
                    .Where(item => item.ObjectivelyValid && item.TakeNumber == takeNumber && string.Equals(item.ExpectedWord, expectedWord, StringComparison.Ordinal))
                    .OrderBy(item => DateTimeOffset.Parse(item.CaptureTimeUtc, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind))
                    .ThenBy(item => Path.GetFileName(item.EvidencePath), StringComparer.Ordinal)
                    .FirstOrDefault();
                if (firstValid == null) break;
                recovered.Add(firstValid);
            }

            return new PhysicalEvidenceResumeReport(fullDirectory, jsonPaths.Length, checksumPaths.Length, verified, recovered);
        }

        static void ValidateSchema(ResumeEvidenceDto dto, string path)
        {
            string name = Path.GetFileName(path);
            if (dto == null) throw new InvalidDataException("evidence JSON root is null: " + name);
            if (!string.Equals(dto.schema_version, Step9CPhysicalCompatibilityConstants.SchemaVersion, StringComparison.Ordinal)) throw Invalid("schema_version", name);
            if (!string.Equals(dto.run_id, Step9CPhysicalCompatibilityConstants.RunId, StringComparison.Ordinal)) throw Invalid("run_id", name);
            if (dto.take_number < 1 || dto.take_number > Step9CPhysicalCompatibilityConstants.FixedWords.Length) throw Invalid("take_number", name);
            if (!string.Equals(dto.expected_word, Step9CPhysicalCompatibilityConstants.FixedWords[dto.take_number - 1], StringComparison.Ordinal)) throw Invalid("fixed_take_order", name);
            string capturePrefix = $"step9c-take-{dto.take_number:D2}-{dto.expected_word.ToLowerInvariant().Replace('_', '-')}-";
            if (string.IsNullOrWhiteSpace(dto.capture_id) || !dto.capture_id.StartsWith(capturePrefix, StringComparison.Ordinal) ||
                !Guid.TryParseExact(dto.capture_id.Substring(capturePrefix.Length), "D", out _)) throw Invalid("capture_id", name);
            string expectedName = $"take_{dto.take_number:D2}_{dto.expected_word}_{dto.capture_id}.json";
            if (!string.Equals(name, expectedName, StringComparison.Ordinal)) throw Invalid("file_name_binding", name);
            if (!DateTimeOffset.TryParse(dto.capture_time_utc, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out _)) throw Invalid("capture_time_utc", name);
            if (string.IsNullOrWhiteSpace(dto.finalization_state)) throw Invalid("finalization_state", name);
            if (dto.objective_invalid_reasons == null) throw Invalid("objective_invalid_reasons", name);
            if (dto.timestamp_summary == null || dto.counts == null || dto.device == null || dto.body_transform == null || dto.frames == null) throw Invalid("required_section", name);
            if (dto.counts.raw_frames < 0 || dto.counts.raw_frames != dto.frames.Length || dto.counts.raw_frames != (dto.timestamp_summary.timestamps?.Length ?? -1)) throw Invalid("raw_frame_count", name);

            int poseFrames = 0;
            int leftFrames = 0;
            int rightFrames = 0;
            int normalizedFrames = 0;
            int normalizedValid = 0;
            int bodyTransformFrames = 0;
            bool strictlyIncreasing = true;
            double previousTimestamp = double.NaN;
            for (int index = 0; index < dto.frames.Length; index++)
            {
                ResumeFrameDto frame = dto.frames[index];
                if (frame == null || frame.frame_index != index || !Finite(frame.timestamp_seconds)) throw Invalid($"frame[{index}]", name);
                if (index > 0 && frame.timestamp_seconds <= previousTimestamp) strictlyIncreasing = false;
                previousTimestamp = frame.timestamp_seconds;
                if (frame.normalized_mask == null || frame.normalized_mask.Length != 16) throw Invalid($"normalized_mask[{index}]", name);
                if (frame.pose_present) poseFrames++;
                if (frame.left_hand_present) leftFrames++;
                if (frame.right_hand_present) rightFrames++;
                if (frame.has_normalized_frame) normalizedFrames++;
                if (frame.body_transform_available) bodyTransformFrames++;
                normalizedValid += frame.normalized_mask.Count(value => value);
                if (dto.timestamp_summary.timestamps[index] != frame.timestamp_seconds) throw Invalid($"timestamp_binding[{index}]", name);
            }
            if (strictlyIncreasing != dto.timestamp_summary.strictly_increasing) throw Invalid("timestamp_monotonicity_summary", name);
            if (poseFrames != dto.counts.pose_present_frames || leftFrames != dto.counts.left_hand_present_frames || rightFrames != dto.counts.right_hand_present_frames) throw Invalid("presence_count", name);
            if (normalizedFrames != dto.counts.normalized_frames || normalizedValid != dto.counts.normalized_valid_landmarks) throw Invalid("normalized_count", name);
            if (bodyTransformFrames != dto.body_transform.available_frames) throw Invalid("body_transform_count", name);

            if (dto.pre_export_objectively_valid)
            {
                if (dto.objective_invalid_reasons.Length != 0) throw Invalid("objective_validity_reasons", name);
                if (HasFailure(dto.failure)) throw Invalid("objective_validity_failure", name);
                if (!string.Equals(dto.finalization_state, "Completed", StringComparison.Ordinal)) throw Invalid("objective_validity_finalization", name);
                if (!dto.device.device_opened) throw Invalid("objective_validity_device", name);
                if (!dto.device.preview_observed) throw Invalid("objective_validity_preview", name);
                if (!dto.device.provider_bound) throw Invalid("objective_validity_provider", name);
                if (!dto.device.provider_readiness_observed) throw Invalid("objective_validity_readiness", name);
                if (dto.counts.raw_frames == 0) throw Invalid("objective_validity_frames", name);
                if (!dto.timestamp_summary.strictly_increasing) throw Invalid("objective_validity_timestamps", name);
                if (dto.counts.pose_present_frames == 0) throw Invalid("objective_validity_pose", name);
                if (dto.counts.left_hand_present_frames + dto.counts.right_hand_present_frames == 0) throw Invalid("objective_validity_hand", name);
                if (dto.body_transform.available_frames == 0) throw Invalid("objective_validity_body_transform", name);
            }
            else if (dto.objective_invalid_reasons.Length == 0)
            {
                throw Invalid("objective_invalid_reasons_empty", name);
            }
        }

        static InvalidDataException Invalid(string field, string name) => new InvalidDataException($"evidence schema validation failed ({field}): {name}");
        static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        static bool HasFailure(ResumeFailureDto failure) => failure != null && !string.IsNullOrWhiteSpace(failure.exception_type);

        [Serializable]
        sealed class ResumeEvidenceDto
        {
            public string schema_version;
            public string run_id;
            public string capture_id;
            public string expected_word;
            public int take_number;
            public string capture_time_utc;
            public ResumeDeviceDto device;
            public ResumeTimestampDto timestamp_summary;
            public ResumeCountsDto counts;
            public ResumeBodyTransformDto body_transform;
            public string finalization_state;
            public bool pre_export_objectively_valid;
            public string[] objective_invalid_reasons;
            public ResumeFailureDto failure;
            public ResumeFrameDto[] frames;
        }

        [Serializable] sealed class ResumeDeviceDto { public bool device_opened; public bool preview_observed; public bool provider_bound; public bool provider_readiness_observed; }
        [Serializable] sealed class ResumeTimestampDto { public bool strictly_increasing; public double[] timestamps; }
        [Serializable] sealed class ResumeCountsDto
        {
            public int raw_frames;
            public int normalized_frames;
            public int pose_present_frames;
            public int left_hand_present_frames;
            public int right_hand_present_frames;
            public int normalized_valid_landmarks;
        }
        [Serializable] sealed class ResumeBodyTransformDto { public int available_frames; }
        [Serializable] sealed class ResumeFailureDto { public string exception_type; public string message; }
        [Serializable] sealed class ResumeFrameDto
        {
            public int frame_index;
            public double timestamp_seconds;
            public bool pose_present;
            public bool left_hand_present;
            public bool right_hand_present;
            public bool body_transform_available;
            public bool has_normalized_frame;
            public bool[] normalized_mask;
        }
    }
}
