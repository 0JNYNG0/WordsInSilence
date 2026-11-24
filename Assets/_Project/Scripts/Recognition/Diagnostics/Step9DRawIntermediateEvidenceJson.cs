using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace WordsInSilence.Recognition.Diagnostics
{
    public sealed class Step9DRawIntermediateEvidenceReceipt
    {
        public string EvidencePath { get; }
        public string ChecksumPath { get; }
        public string Sha256 { get; }
        public long SizeBytes { get; }
        public bool SchemaValidated { get; }
        public bool ChecksumVerified { get; }

        internal Step9DRawIntermediateEvidenceReceipt(
            string evidencePath,
            string checksumPath,
            string sha256,
            long sizeBytes,
            bool schemaValidated,
            bool checksumVerified)
        {
            EvidencePath = evidencePath;
            ChecksumPath = checksumPath;
            Sha256 = sha256;
            SizeBytes = sizeBytes;
            SchemaValidated = schemaValidated;
            ChecksumVerified = checksumVerified;
        }
    }

    public static class Step9DRawIntermediateEvidenceJson
    {
        static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(false, true);

        public static string Serialize(Step9DRawIntermediateDiagnosticSnapshot snapshot)
        {
            ValidateSnapshot(snapshot);
            Step9DRawIntermediatePreprocessingTrace trace = snapshot.Trace;
            Step9DRawIntermediateFrameIdentity[] identities = snapshot.CopyFrameIdentities();
            var output = new StringBuilder(Math.Max(4096, trace.RawFrameCount * 10000));
            output.Append('{');
            Property(output, "schema_version", trace.SchemaVersion); Comma(output);
            Property(output, "run_id", snapshot.RunId); Comma(output);
            Property(output, "capture_id", snapshot.CaptureId); Comma(output);
            Property(output, "expected_word", snapshot.ExpectedWord); Comma(output);
            output.Append("\"capture_metadata\":{");
            Property(output, "coordinate_convention_id", trace.CoordinateConventionId); Comma(output);
            Property(output, "coordinate_space", trace.CoordinateSpace.ToString()); Comma(output);
            Property(output, "axis_and_depth", trace.AxisAndDepth); Comma(output);
            Property(output, "handedness", trace.Handedness); Comma(output);
            Property(output, "mirror", trace.Mirror); Comma(output);
            Property(output, "selected_max_gap", trace.SelectedMaxGap); Comma(output);
            Property(output, "timestamp_discontinuity_seconds", trace.TimestampDiscontinuitySeconds); Comma(output);
            Property(output, "raw_frame_count", trace.RawFrameCount); Comma(output);
            Property(output, "normalized_frame_count", trace.NormalizedFrameCount); Comma(output);
            output.Append("\"raw_landmark_ordering\":"); WriteStringArray(output, trace.CopyRawLandmarkOrdering()); Comma(output);
            output.Append("\"normalized_landmark_ordering\":"); WriteStringArray(output, trace.CopyNormalizedLandmarkOrdering());
            Step9DInferenceImageGeometry geometry = snapshot.InferenceImageGeometry;
            if (geometry != null)
            {
                Comma(output);
                output.Append("\"inference_image_geometry\":{");
                Property(output, "webcam_texture_width", geometry.WebCamTextureWidth); Comma(output);
                Property(output, "webcam_texture_height", geometry.WebCamTextureHeight); Comma(output);
                Property(output, "video_rotation_angle", geometry.VideoRotationAngle); Comma(output);
                output.Append("\"video_vertically_mirrored\":"); output.Append(geometry.VideoVerticallyMirrored ? "true" : "false"); Comma(output);
                Property(output, "inference_texture_width", geometry.InferenceTextureWidth); Comma(output);
                Property(output, "inference_texture_height", geometry.InferenceTextureHeight);
                output.Append('}');
            }
            Step9DMergedInferenceGeometry captureGeom = snapshot.CaptureInferenceGeometry;
            if (captureGeom != null)
            {
                Comma(output);
                output.Append("\"inference_frame_geometry\":{");
                output.Append("\"both_present\":"); output.Append(captureGeom.BothPresent ? "true" : "false"); Comma(output);
                output.Append("\"both_available\":"); output.Append(captureGeom.BothAvailable ? "true" : "false"); Comma(output);
                output.Append("\"dimensionally_compatible\":"); output.Append(captureGeom.DimensionallyCompatible ? "true" : "false"); Comma(output);
                output.Append("\"full_contract_compatible\":"); output.Append(captureGeom.FullContractCompatible ? "true" : "false"); Comma(output);
                output.Append("\"first_mismatch_description\":"); NullableString(output, captureGeom.FirstMismatchDescription); Comma(output);
                output.Append("\"hand_pose_timestamp_skew_ms\":"); NullableLong(output, captureGeom.HandPoseTimestampSkewMs);
                WriteAcquisitionResult(output, "hand_acquisition", captureGeom.HandResult);
                WriteAcquisitionResult(output, "pose_acquisition", captureGeom.PoseResult);
                if (captureGeom.Hand != null) { Comma(output); output.Append("\"hand\":"); WriteFrameGeometry(output, captureGeom.Hand); }
                if (captureGeom.Pose != null) { Comma(output); output.Append("\"pose\":"); WriteFrameGeometry(output, captureGeom.Pose); }
                output.Append('}');
            }
            Step9DCaptureGeometryConsistencySummary consistencySummary = snapshot.GeometryConsistencySummary;
            if (consistencySummary != null)
            {
                Comma(output);
                output.Append("\"geometry_consistency_summary\":{");
                Property(output, "checked_count", consistencySummary.CheckedCount); Comma(output);
                output.Append("\"first_hand_timestamp_ms\":"); output.Append(consistencySummary.FirstHandTimestampMs.ToString(CultureInfo.InvariantCulture)); Comma(output);
                output.Append("\"last_hand_timestamp_ms\":"); output.Append(consistencySummary.LastHandTimestampMs.ToString(CultureInfo.InvariantCulture)); Comma(output);
                output.Append("\"first_pose_timestamp_ms\":"); output.Append(consistencySummary.FirstPoseTimestampMs.ToString(CultureInfo.InvariantCulture)); Comma(output);
                output.Append("\"last_pose_timestamp_ms\":"); output.Append(consistencySummary.LastPoseTimestampMs.ToString(CultureInfo.InvariantCulture)); Comma(output);
                output.Append("\"max_hand_pose_abs_skew_ms\":"); output.Append(consistencySummary.MaxHandPoseAbsSkewMs.ToString(CultureInfo.InvariantCulture));
                output.Append('}');
            }
            output.Append("},");

            output.Append("\"transient_hand_identity_drops\":[");
            IReadOnlyList<Step9DAmbiguityDropRecord> drops = snapshot.TransientHandIdentityDrops;
            if (drops != null)
            {
                for (int dropIndex = 0; dropIndex < drops.Count; dropIndex++)
                {
                    if (dropIndex > 0) Comma(output);
                    Step9DAmbiguityDropRecord drop = drops[dropIndex];
                    output.Append('{');
                    Property(output, "capture_id", drop.CaptureId); Comma(output);
                    Property(output, "drop_ordinal", drop.DropOrdinal); Comma(output);
                    Property(output, "provider_frame_index", drop.ProviderFrameIndex); Comma(output);
                    Property(output, "provider_timestamp_seconds", drop.ProviderTimestampSeconds); Comma(output);
                    Property(output, "category_0", drop.Category0); Comma(output);
                    Property(output, "score_0", drop.Score0); Comma(output);
                    Property(output, "category_1", drop.Category1); Comma(output);
                    Property(output, "score_1", drop.Score1); Comma(output);
                    Property(output, "reason", drop.Reason);
                    output.Append('}');
                }
            }
            output.Append("],");

            output.Append("\"frame_identity\":[");
            for (int frame = 0; frame < identities.Length; frame++)
            {
                if (frame > 0) Comma(output);
                Step9DRawIntermediateFrameIdentity identity = identities[frame];
                output.Append('{');
                Property(output, "capture_local_frame_index", identity.CaptureLocalFrameIndex); Comma(output);
                Property(output, "observation_timestamp_seconds", identity.ObservationTimestampSeconds); Comma(output);
                output.Append("\"provider_frame_index\":"); NullableLong(output, identity.ProviderFrameIndex); Comma(output);
                output.Append("\"bridge_observation_index\":"); NullableLong(output, identity.BridgeObservationIndex); Comma(output);
                output.Append("\"lifetime_provider_frame_index\":"); NullableLong(output, identity.LifetimeProviderFrameIndex);
                output.Append('}');
            }
            output.Append("],");

            output.Append("\"raw_and_preprocessing\":{");
            output.Append("\"raw_coordinates_shape\":"); WriteIntArray(output, trace.RawCoordinatesShape); Comma(output);
            output.Append("\"raw_coordinates\":"); WriteDoubleTensor3(output, trace.CopyRawCoordinates(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"raw_original_valid_mask_shape\":"); WriteIntArray(output, trace.RawMaskShape); Comma(output);
            output.Append("\"raw_original_valid_mask\":"); WriteBoolMatrix(output, trace.CopyRawOriginalValidMask(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"raw_reappearance_mask\":"); WriteBoolMatrix(output, trace.CopyRawReappearanceMask(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"raw_strong_depth_mask\":"); WriteBoolMatrix(output, trace.CopyRawStrongDepthMask(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"timestamp_discontinuity_boundary_mask\":"); WriteBoolMatrix(output, trace.CopyTimestampBoundaryMask(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"combined_boundary_mask\":"); WriteBoolMatrix(output, trace.CopyCombinedBoundaryMask(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"post_gap_interpolation_coordinates_shape\":"); WriteIntArray(output, trace.RawCoordinatesShape); Comma(output);
            output.Append("\"post_gap_interpolation_coordinates\":"); WriteDoubleTensor3(output, trace.CopyPostGapInterpolationCoordinates(), trace.RawFrameCount, 75); Comma(output);
            output.Append("\"interpolation_mask\":"); WriteBoolMatrix(output, trace.CopyInterpolationMask(), trace.RawFrameCount, 75);
            output.Append("},");

            output.Append("\"torso_transform\":{");
            output.Append("\"vector_shape\":"); WriteIntArray(output, trace.TorsoVectorShape); Comma(output);
            output.Append("\"source_valid\":"); WriteBoolArray(output, trace.CopyTorsoSourceValid()); Comma(output);
            output.Append("\"final_valid\":"); WriteBoolArray(output, trace.CopyTorsoFinalValid()); Comma(output);
            output.Append("\"provenance\":"); WriteProvenanceArray(output, trace.CopyTorsoProvenance()); Comma(output);
            output.Append("\"origin\":"); WriteDoubleMatrix(output, trace.CopyTorsoOrigin(), trace.RawFrameCount, 3); Comma(output);
            output.Append("\"right_axis\":"); WriteDoubleMatrix(output, trace.CopyTorsoRight(), trace.RawFrameCount, 3); Comma(output);
            output.Append("\"up_axis\":"); WriteDoubleMatrix(output, trace.CopyTorsoUp(), trace.RawFrameCount, 3); Comma(output);
            output.Append("\"forward_axis\":"); WriteDoubleMatrix(output, trace.CopyTorsoForward(), trace.RawFrameCount, 3); Comma(output);
            output.Append("\"shoulder_width_scale\":"); WriteDoubleArray(output, trace.CopyTorsoShoulderWidthScale());
            output.Append("},");

            output.Append("\"normalized_output\":{");
            output.Append("\"coordinates_shape\":"); WriteIntArray(output, trace.NormalizedCoordinatesShape); Comma(output);
            output.Append("\"coordinates\":"); WriteDoubleTensor3(output, trace.CopyNormalizedCoordinates(), trace.NormalizedFrameCount, 16); Comma(output);
            output.Append("\"mask_shape\":"); WriteIntArray(output, trace.NormalizedMaskShape); Comma(output);
            output.Append("\"mask\":"); WriteBoolMatrix(output, trace.CopyNormalizedMask(), trace.NormalizedFrameCount, 16); Comma(output);
            Property(output, "valid_landmark_count", trace.ValidLandmarkCount); Comma(output);
            Property(output, "interpolated_landmark_count", trace.InterpolatedLandmarkCount);
            output.Append("}}");
            output.Append('\n');
            return output.ToString();
        }

        public static Step9DRawIntermediateEvidenceReceipt Export(
            string path,
            Step9DRawIntermediateDiagnosticSnapshot snapshot)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("evidence path is required", nameof(path));
            string fullPath = Path.GetFullPath(path);
            string checksumPath = fullPath + ".sha256";
            string temporaryPath = fullPath + ".tmp";
            string temporaryChecksumPath = checksumPath + ".tmp";
            if (File.Exists(fullPath) || File.Exists(checksumPath) || File.Exists(temporaryPath) || File.Exists(temporaryChecksumPath))
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.OUTPUT_ALREADY_EXISTS,
                    "diagnostic evidence export never overwrites target, checksum or temporary files");
            string directory = Path.GetDirectoryName(fullPath);
            if (string.IsNullOrWhiteSpace(directory))
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.SCHEMA_VALIDATION_FAILED,
                    "diagnostic evidence directory is invalid");
            Directory.CreateDirectory(directory);

            byte[] bytes = Utf8NoBom.GetBytes(Serialize(snapshot));
            if (!ValidateSerializedBytes(bytes))
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.SCHEMA_VALIDATION_FAILED,
                    "serialized diagnostic evidence failed schema token validation");
            WriteNew(temporaryPath, bytes);
            File.Move(temporaryPath, fullPath);
            string hash = Sha256(bytes);
            byte[] checksumBytes = Utf8NoBom.GetBytes(hash + "  " + Path.GetFileName(fullPath) + "\n");
            WriteNew(temporaryChecksumPath, checksumBytes);
            File.Move(temporaryChecksumPath, checksumPath);
            bool verified = Verify(fullPath, checksumPath);
            if (!verified)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.CHECKSUM_VERIFICATION_FAILED,
                    "diagnostic evidence detached checksum verification failed");
            return new Step9DRawIntermediateEvidenceReceipt(fullPath, checksumPath, hash, bytes.LongLength, true, true);
        }

        public static bool Verify(string evidencePath, string checksumPath)
        {
            try
            {
                if (!File.Exists(evidencePath) || !File.Exists(checksumPath)) return false;
                byte[] evidence = File.ReadAllBytes(evidencePath);
                if (!ValidateSerializedBytes(evidence)) return false;
                string line = File.ReadAllText(checksumPath, Utf8NoBom);
                string fileName = Path.GetFileName(evidencePath);
                if (line.Length != 64 + 2 + fileName.Length + 1 || line[64] != ' ' || line[65] != ' ' || line[line.Length - 1] != '\n') return false;
                string expected = line.Substring(0, 64);
                for (int index = 0; index < expected.Length; index++)
                {
                    char value = expected[index];
                    if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) return false;
                }
                if (!string.Equals(line.Substring(66, fileName.Length), fileName, StringComparison.Ordinal)) return false;
                return string.Equals(expected, Sha256(evidence), StringComparison.Ordinal);
            }
            catch
            {
                return false;
            }
        }

        public static void ValidateSnapshot(Step9DRawIntermediateDiagnosticSnapshot snapshot)
        {
            if (snapshot == null) throw new ArgumentNullException(nameof(snapshot));
            Step9DRawIntermediatePreprocessingTrace trace = snapshot.Trace ?? throw new ArgumentException("trace is required", nameof(snapshot));
            if (!string.Equals(trace.SchemaVersion, Step9DRawIntermediateDiagnosticContract.SchemaVersion, StringComparison.Ordinal))
                throw Failure("trace schema version is invalid");
            if (snapshot.ObservationCount != trace.RawFrameCount || trace.RawFrameCount != trace.NormalizedFrameCount)
                throw Failure("raw, normalized and observation frame counts differ");
            if (snapshot.Result.RawFrameCount != trace.RawFrameCount || snapshot.Result.NormalizedFrameCount != trace.NormalizedFrameCount ||
                snapshot.Result.ValidLandmarkCount != trace.ValidLandmarkCount || snapshot.Result.InterpolatedLandmarkCount != trace.InterpolatedLandmarkCount)
                throw Failure("shared-core result aggregate values differ from trace");

            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength("raw_coordinates", trace.CopyRawCoordinates().Length, trace.RawFrameCount, 75, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength("raw_original_valid", trace.CopyRawOriginalValidMask().Length, trace.RawFrameCount, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength("post_gap_coordinates", trace.CopyPostGapInterpolationCoordinates().Length, trace.RawFrameCount, 75, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength("normalized_coordinates", trace.CopyNormalizedCoordinates().Length, trace.NormalizedFrameCount, 16, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength("normalized_mask", trace.CopyNormalizedMask().Length, trace.NormalizedFrameCount, 16);

            bool[] reappearance = trace.CopyRawReappearanceMask();
            bool[] strong = trace.CopyRawStrongDepthMask();
            bool[] timestamp = trace.CopyTimestampBoundaryMask();
            bool[] combined = trace.CopyCombinedBoundaryMask();
            for (int index = 0; index < combined.Length; index++)
                if (combined[index] != (reappearance[index] || strong[index] || timestamp[index]))
                    throw Failure($"combined boundary differs at flat index {index}");

            bool[] sourceValid = trace.CopyTorsoSourceValid();
            bool[] finalValid = trace.CopyTorsoFinalValid();
            Step9DTorsoTransformProvenance[] provenance = trace.CopyTorsoProvenance();
            for (int frame = 0; frame < trace.RawFrameCount; frame++)
            {
                bool valid = provenance[frame] == Step9DTorsoTransformProvenance.DIRECT
                    ? sourceValid[frame] && finalValid[frame]
                    : provenance[frame] == Step9DTorsoTransformProvenance.UNAVAILABLE
                        ? !sourceValid[frame] && !finalValid[frame]
                        : !sourceValid[frame] && finalValid[frame];
                if (!valid) throw Failure($"torso provenance is inconsistent at frame {frame}");
            }

            Step9DRawIntermediateFrameIdentity[] identities = snapshot.CopyFrameIdentities();
            double[] normalized = trace.CopyNormalizedCoordinates();
            bool[] normalizedMask = trace.CopyNormalizedMask();
            int countedValid = 0;
            for (int frame = 0; frame < trace.NormalizedFrameCount; frame++)
            {
                if (identities[frame].CaptureLocalFrameIndex != frame)
                    throw Failure($"capture-local frame identity differs at frame {frame}");
                if (identities[frame].ObservationTimestampSeconds != snapshot.Result.Frames[frame].Timestamp)
                    throw Failure($"timestamp differs at frame {frame}");
                double[] resultCoordinates = snapshot.Result.Frames[frame].CopyCoordinates();
                bool[] resultMask = snapshot.Result.Frames[frame].CopyValidMask();
                for (int landmark = 0; landmark < 16; landmark++)
                {
                    int maskIndex = frame * 16 + landmark;
                    if (resultMask[landmark] != normalizedMask[maskIndex]) throw Failure($"normalized mask differs at frame {frame}, landmark {landmark}");
                    if (normalizedMask[maskIndex]) countedValid++;
                    for (int axis = 0; axis < 3; axis++)
                    {
                        int coordinateIndex = maskIndex * 3 + axis;
                        if (!DoubleExact(resultCoordinates[landmark * 3 + axis], normalized[coordinateIndex]))
                            throw Failure($"normalized coordinate differs at frame {frame}, landmark {landmark}, axis {axis}");
                    }
                }
            }
            if (countedValid != trace.ValidLandmarkCount) throw Failure("normalized valid aggregate differs from mask");
        }

        static bool ValidateSerializedBytes(byte[] bytes)
        {
            string text = Utf8NoBom.GetString(bytes);
            return text.EndsWith("\n", StringComparison.Ordinal) &&
                text.StartsWith("{\"schema_version\":\"" + Step9DRawIntermediateDiagnosticContract.SchemaVersion + "\"", StringComparison.Ordinal) &&
                text.Contains("\"raw_coordinates_shape\":[", StringComparison.Ordinal) &&
                text.Contains("\"torso_transform\":{", StringComparison.Ordinal) &&
                text.Contains("\"normalized_output\":{", StringComparison.Ordinal) &&
                !text.Contains(":NaN", StringComparison.Ordinal) &&
                !text.Contains(":Infinity", StringComparison.Ordinal) &&
                !text.Contains(":-Infinity", StringComparison.Ordinal);
        }

        static void WriteNew(string path, byte[] bytes)
        {
            using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                stream.Write(bytes, 0, bytes.Length);
        }

        static string Sha256(byte[] bytes)
        {
            using (var algorithm = SHA256.Create())
            {
                byte[] hash = algorithm.ComputeHash(bytes);
                var output = new StringBuilder(hash.Length * 2);
                foreach (byte value in hash) output.Append(value.ToString("x2", CultureInfo.InvariantCulture));
                return output.ToString();
            }
        }

        static void WriteAcquisitionResult(StringBuilder output, string fieldName, Step9DGeometryAcquisitionResult result)
        {
            Comma(output);
            output.Append('"'); output.Append(fieldName); output.Append("\":{");
            if (result == null)
            {
                Property(output, "status", "null");
                output.Append('}');
                return;
            }
            Property(output, "status", result.Status.ToString()); Comma(output);
            Property(output, "diagnostic_code", result.DiagnosticCode);
            output.Append('}');
        }

        static void WriteFrameGeometry(StringBuilder output, Step9DInferenceFrameGeometry g)
        {
            output.Append('{');
            Property(output, "provenance", g.Provenance.ToString()); Comma(output);
            output.Append("\"requested_width\":"); NullableInt(output, g.RequestedWidthOrNull); Comma(output);
            output.Append("\"requested_height\":"); NullableInt(output, g.RequestedHeightOrNull); Comma(output);
            output.Append("\"requested_fps\":"); NullableInt(output, g.RequestedFpsOrNull); Comma(output);
            Property(output, "observed_webcam_width", g.ObservedWebcamWidth); Comma(output);
            Property(output, "observed_webcam_height", g.ObservedWebcamHeight); Comma(output);
            output.Append("\"observed_video_rotation_angle\":"); NullableInt(output, g.ObservedVideoRotationAngleOrNull); Comma(output);
            output.Append("\"observed_video_vertically_mirrored\":"); NullableBool(output, g.ObservedVideoVerticallyMirroredOrNull); Comma(output);
            Property(output, "inference_texture_width", g.InferenceTextureWidth); Comma(output);
            Property(output, "inference_texture_height", g.InferenceTextureHeight); Comma(output);
            Property(output, "inference_image_width", g.InferenceImageWidth); Comma(output);
            Property(output, "inference_image_height", g.InferenceImageHeight); Comma(output);
            Property(output, "image_format", g.ImageFormat); Comma(output);
            Property(output, "pre_inference_horizontal_flip", g.PreInferenceHorizontalFlip.ToString()); Comma(output);
            Property(output, "pre_inference_vertical_flip", g.PreInferenceVerticalFlip.ToString()); Comma(output);
            output.Append("\"pre_inference_rotation_degrees\":"); NullableInt(output, g.PreInferenceRotationDegreesOrNull); Comma(output);
            Property(output, "crop", g.Crop.ToString()); Comma(output);
            Property(output, "resize", g.Resize.ToString()); Comma(output);
            Property(output, "letterbox", g.Letterbox.ToString()); Comma(output);
            output.Append("\"provider_frame_index\":"); NullableLong(output, g.ProviderFrameIndexOrNull); Comma(output);
            output.Append("\"mediapipe_timestamp_ms\":"); output.Append(g.MediaPipeTimestampMs.ToString(CultureInfo.InvariantCulture)); Comma(output);
            output.Append("\"unity_frame_count\":"); NullableInt(output, g.UnityFrameCountOrNull); Comma(output);
            Property(output, "pixel_buffer_row_order", g.PixelBufferRowOrder);
            output.Append('}');
        }

        static void NullableInt(StringBuilder output, int? value)
        {
            if (value.HasValue) output.Append(value.Value.ToString(CultureInfo.InvariantCulture));
            else output.Append("null");
        }

        static void NullableBool(StringBuilder output, bool? value)
        {
            if (value.HasValue) output.Append(value.Value ? "true" : "false");
            else output.Append("null");
        }

        static void NullableString(StringBuilder output, string value)
        {
            if (value == null) { output.Append("null"); return; }
            String(output, value);
        }

        static void WriteDoubleTensor3(StringBuilder output, double[] values, int frames, int landmarks)
        {
            output.Append('[');
            for (int frame = 0; frame < frames; frame++)
            {
                if (frame > 0) Comma(output);
                output.Append('[');
                for (int landmark = 0; landmark < landmarks; landmark++)
                {
                    if (landmark > 0) Comma(output);
                    output.Append('[');
                    for (int axis = 0; axis < 3; axis++)
                    {
                        if (axis > 0) Comma(output);
                        NumberOrNull(output, values[(frame * landmarks + landmark) * 3 + axis]);
                    }
                    output.Append(']');
                }
                output.Append(']');
            }
            output.Append(']');
        }

        static void WriteBoolMatrix(StringBuilder output, bool[] values, int rows, int columns)
        {
            output.Append('[');
            for (int row = 0; row < rows; row++)
            {
                if (row > 0) Comma(output);
                output.Append('[');
                for (int column = 0; column < columns; column++)
                {
                    if (column > 0) Comma(output);
                    output.Append(values[row * columns + column] ? "true" : "false");
                }
                output.Append(']');
            }
            output.Append(']');
        }

        static void WriteDoubleMatrix(StringBuilder output, double[] values, int rows, int columns)
        {
            output.Append('[');
            for (int row = 0; row < rows; row++)
            {
                if (row > 0) Comma(output);
                output.Append('[');
                for (int column = 0; column < columns; column++)
                {
                    if (column > 0) Comma(output);
                    NumberOrNull(output, values[row * columns + column]);
                }
                output.Append(']');
            }
            output.Append(']');
        }

        static void WriteDoubleArray(StringBuilder output, double[] values)
        {
            output.Append('[');
            for (int index = 0; index < values.Length; index++)
            {
                if (index > 0) Comma(output);
                NumberOrNull(output, values[index]);
            }
            output.Append(']');
        }

        static void WriteBoolArray(StringBuilder output, bool[] values)
        {
            output.Append('[');
            for (int index = 0; index < values.Length; index++)
            {
                if (index > 0) Comma(output);
                output.Append(values[index] ? "true" : "false");
            }
            output.Append(']');
        }

        static void WriteProvenanceArray(StringBuilder output, Step9DTorsoTransformProvenance[] values)
        {
            output.Append('[');
            for (int index = 0; index < values.Length; index++)
            {
                if (index > 0) Comma(output);
                String(output, values[index].ToString());
            }
            output.Append(']');
        }

        static void WriteStringArray(StringBuilder output, IReadOnlyList<string> values)
        {
            output.Append('[');
            for (int index = 0; index < values.Count; index++)
            {
                if (index > 0) Comma(output);
                String(output, values[index]);
            }
            output.Append(']');
        }

        static void WriteIntArray(StringBuilder output, int[] values)
        {
            output.Append('[');
            for (int index = 0; index < values.Length; index++)
            {
                if (index > 0) Comma(output);
                output.Append(values[index].ToString(CultureInfo.InvariantCulture));
            }
            output.Append(']');
        }

        static void NullableLong(StringBuilder output, long? value)
        {
            if (value.HasValue) output.Append(value.Value.ToString(CultureInfo.InvariantCulture));
            else output.Append("null");
        }

        static void Property(StringBuilder output, string name, string value) { String(output, name); output.Append(':'); String(output, value); }
        static void Property(StringBuilder output, string name, int value) { String(output, name); output.Append(':').Append(value.ToString(CultureInfo.InvariantCulture)); }
        static void Property(StringBuilder output, string name, double value) { String(output, name); output.Append(':'); NumberOrNull(output, value); }
        static void NumberOrNull(StringBuilder output, double value)
        {
            Step9DRawIntermediateDiagnosticContract.RejectInfinity("serialized number", value);
            output.Append(double.IsNaN(value) ? "null" : value.ToString("R", CultureInfo.InvariantCulture));
        }
        static bool DoubleExact(double left, double right) => (double.IsNaN(left) && double.IsNaN(right)) || left.Equals(right);
        static void Comma(StringBuilder output) => output.Append(',');

        static void String(StringBuilder output, string value)
        {
            output.Append('"');
            if (value != null)
            {
                foreach (char character in value)
                {
                    switch (character)
                    {
                        case '"': output.Append("\\\""); break;
                        case '\\': output.Append("\\\\"); break;
                        case '\b': output.Append("\\b"); break;
                        case '\f': output.Append("\\f"); break;
                        case '\n': output.Append("\\n"); break;
                        case '\r': output.Append("\\r"); break;
                        case '\t': output.Append("\\t"); break;
                        default:
                            if (character < 0x20) output.Append("\\u").Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                            else output.Append(character);
                            break;
                    }
                }
            }
            output.Append('"');
        }

        static Step9DRawIntermediateDiagnosticException Failure(string message) =>
            new Step9DRawIntermediateDiagnosticException(
                Step9DRawIntermediateDiagnosticFailureCode.SCHEMA_VALIDATION_FAILED,
                message);
    }
}
