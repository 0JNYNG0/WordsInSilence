using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    public static class PhysicalCaptureEvidenceSchema
    {
        public static IReadOnlyList<string> Validate(PhysicalCaptureEvidence evidence)
        {
            var errors = new List<string>();
            if (evidence == null) { errors.Add("evidence_is_null"); return errors.AsReadOnly(); }
            if (!string.Equals(evidence.SchemaVersion, Step9CPhysicalCompatibilityConstants.SchemaVersion, StringComparison.Ordinal)) errors.Add("schema_version");
            if (!string.Equals(evidence.RunId, Step9CPhysicalCompatibilityConstants.RunId, StringComparison.Ordinal)) errors.Add("run_id");
            if (evidence.TakeNumber < 1 || evidence.TakeNumber > 3) errors.Add("take_number");
            else if (!string.Equals(evidence.ExpectedWord, Step9CPhysicalCompatibilityConstants.FixedWords[evidence.TakeNumber - 1], StringComparison.Ordinal)) errors.Add("fixed_take_order");
            if (string.IsNullOrWhiteSpace(evidence.CaptureId)) errors.Add("capture_id");
            if (!DateTimeOffset.TryParse(evidence.CaptureTimeUtc, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out _)) errors.Add("capture_time_utc");
            if (!DateTimeOffset.TryParse(evidence.CaptureTimeLocal, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out _)) errors.Add("capture_time_local");
            if (!string.Equals(evidence.MediaPipePackageVersion, Step9CPhysicalCompatibilityConstants.MediaPipePackageVersion, StringComparison.Ordinal)) errors.Add("mediapipe_version");
            if (evidence.RawFrameCount != evidence.Frames.Count) errors.Add("raw_frame_count");
            if (evidence.Device.ProviderFrameCount < evidence.Device.BridgeObservationCount) errors.Add("provider_frame_count");
            if (evidence.CaptureWindow != null)
            {
                PhysicalCaptureWindowSnapshot window = evidence.CaptureWindow;
                if (window.ProviderFrameCount != evidence.Device.ProviderFrameCount ||
                    window.BridgeObservationCount != evidence.Device.BridgeObservationCount) errors.Add("capture_window_counts");
                if (window.ProviderFrameCount == 0)
                {
                    if (window.FirstCaptureProviderFrameIndex != -1 || window.FinalCaptureProviderFrameIndex != -1 ||
                        window.FirstCaptureLifetimeProviderFrameIndex != -1 || window.FinalCaptureLifetimeProviderFrameIndex != -1)
                        errors.Add("capture_window_empty_range");
                }
                else
                {
                    if (window.FirstCaptureProviderFrameIndex != 0 || window.FinalCaptureProviderFrameIndex != window.ProviderFrameCount - 1)
                        errors.Add("capture_window_local_range");
                    if (window.FirstCaptureLifetimeProviderFrameIndex != window.StartNextLifetimeProviderFrameIndex ||
                        window.FinalCaptureLifetimeProviderFrameIndex - window.FirstCaptureLifetimeProviderFrameIndex + 1 != window.ProviderFrameCount)
                        errors.Add("capture_window_lifetime_range");
                    if (!IsFinite(window.FirstCaptureProviderTimestampSeconds) || !IsFinite(window.FinalCaptureProviderTimestampSeconds) ||
                        window.FirstCaptureProviderTimestampSeconds > window.FinalCaptureProviderTimestampSeconds)
                        errors.Add("capture_window_timestamps");
                }
            }
            if (evidence.DuplicateHandednessFrameCount != evidence.HandIdentityDiagnostics.Count) errors.Add("duplicate_handedness_count");
            int computedResolvedDuplicates = 0;
            for (int index = 0; index < evidence.HandIdentityDiagnostics.Count; index++)
            {
                Step9CHandIdentityDiagnostic diagnostic = evidence.HandIdentityDiagnostics[index];
                if (diagnostic == null) { errors.Add($"hand_identity_diagnostic[{index}]"); continue; }
                if (diagnostic.ProviderFrameIndex < 0 || diagnostic.ProviderFrameIndex >= evidence.Device.ProviderFrameCount) errors.Add($"hand_identity_frame_index[{index}]");
                if (!IsFinite(diagnostic.ProviderTimestampSeconds) || diagnostic.ProviderTimestampSeconds < 0.0) errors.Add($"hand_identity_timestamp[{index}]");
                if (diagnostic.SourceHandCount < 2) errors.Add($"hand_identity_source_count[{index}]");
                if (!IsCategory(diagnostic.SourceCategory0) || !IsCategory(diagnostic.SourceCategory1) ||
                    !string.Equals(diagnostic.SourceCategory0, diagnostic.SourceCategory1, StringComparison.Ordinal)) errors.Add($"hand_identity_categories[{index}]");
                if (!IsScore(diagnostic.SourceScore0) || !IsScore(diagnostic.SourceScore1)) errors.Add($"hand_identity_scores[{index}]");
                if (diagnostic.Outcome == Step9CHandIdentityOutcome.ResolvedByPoseWrist) computedResolvedDuplicates++;
                if (string.IsNullOrWhiteSpace(diagnostic.Reason)) errors.Add($"hand_identity_reason[{index}]");
            }
            if (computedResolvedDuplicates != evidence.ResolvedDuplicateHandednessFrameCount ||
                evidence.AmbiguousDuplicateHandednessFrameCount != evidence.HandIdentityDiagnostics.Count - computedResolvedDuplicates)
                errors.Add("hand_identity_outcome_counts");

            int normalizedFrames = 0;
            int normalizedValid = 0;
            double previousTimestamp = double.NaN;
            bool computedStrict = true;
            for (int frameIndex = 0; frameIndex < evidence.Frames.Count; frameIndex++)
            {
                PhysicalFrameEvidence frame = evidence.Frames[frameIndex];
                if (!IsFinite(frame.TimestampSeconds) || frame.TimestampSeconds < 0.0) errors.Add($"timestamp[{frameIndex}]");
                if (frameIndex > 0 && frame.TimestampSeconds <= previousTimestamp) computedStrict = false;
                previousTimestamp = frame.TimestampSeconds;
                bool[] mask = frame.CopyNormalizedMask();
                double[] coordinates = frame.CopyNormalizedCoordinates();
                if (mask.Length != 16 || coordinates.Length != 48) { errors.Add($"normalized_shape[{frameIndex}]"); continue; }
                if (frame.HasNormalizedFrame) normalizedFrames++;
                for (int landmark = 0; landmark < 16; landmark++)
                {
                    bool allFinite = true;
                    bool allNaN = true;
                    for (int axis = 0; axis < 3; axis++)
                    {
                        double value = coordinates[landmark * 3 + axis];
                        allFinite &= IsFinite(value);
                        allNaN &= double.IsNaN(value);
                    }
                    if (mask[landmark])
                    {
                        normalizedValid++;
                        if (!frame.HasNormalizedFrame || !allFinite) errors.Add($"valid_coordinate[{frameIndex},{landmark}]");
                    }
                    else if (!allNaN) errors.Add($"invalid_not_nan[{frameIndex},{landmark}]");
                }
            }
            if (computedStrict != evidence.TimestampsStrictlyIncreasing) errors.Add("timestamp_monotonicity_summary");
            if (normalizedFrames != evidence.NormalizedFrameCount) errors.Add("normalized_frame_count");
            if (normalizedValid != evidence.NormalizedValidLandmarkCount) errors.Add("normalized_valid_count");
            if (evidence.Recognition != null)
            {
                if (evidence.Recognition.TemplateDistances.Count != 9) errors.Add("template_distance_count");
                if (evidence.Recognition.Threshold != Step9CPhysicalCompatibilityConstants.FrozenThreshold) errors.Add("frozen_threshold");
                if (evidence.Recognition.Accepted != (evidence.Recognition.BestDistance <= evidence.Recognition.Threshold)) errors.Add("inclusive_gate_rule");
            }
            if (string.IsNullOrWhiteSpace(evidence.FinalizationState)) errors.Add("finalization_state");
            return errors.AsReadOnly();
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        static bool IsCategory(string value) => string.Equals(value, "Left", StringComparison.Ordinal) || string.Equals(value, "Right", StringComparison.Ordinal);
        static bool IsScore(double value) => IsFinite(value) && value >= 0.0 && value <= 1.0;
    }

    public static class PhysicalCaptureEvidenceJson
    {
        public static string Serialize(PhysicalCaptureEvidence value)
        {
            IReadOnlyList<string> errors = PhysicalCaptureEvidenceSchema.Validate(value);
            if (errors.Count != 0) throw new InvalidDataException("evidence schema validation failed: " + string.Join(",", errors));
            var output = new StringBuilder(32768);
            output.Append('{');
            Property(output, "schema_version", value.SchemaVersion); Comma(output);
            Property(output, "run_id", value.RunId); Comma(output);
            Property(output, "capture_id", value.CaptureId); Comma(output);
            Property(output, "expected_word", value.ExpectedWord); Comma(output);
            Property(output, "take_number", value.TakeNumber); Comma(output);
            Property(output, "capture_time_utc", value.CaptureTimeUtc); Comma(output);
            Property(output, "capture_time_local", value.CaptureTimeLocal); Comma(output);
            output.Append("\"environment\":{");
            Property(output, "unity_version", value.UnityVersion); Comma(output);
            Property(output, "mediapipe_package_version", value.MediaPipePackageVersion); Comma(output);
            Property(output, "scene_name", value.SceneName);
            output.Append("},");
            output.Append("\"device\":{");
            Property(output, "device_name", value.Device.DeviceName); Comma(output);
            Property(output, "requested_width", value.Device.RequestedWidth); Comma(output);
            Property(output, "requested_height", value.Device.RequestedHeight); Comma(output);
            Property(output, "requested_frame_rate", value.Device.RequestedFrameRate); Comma(output);
            Property(output, "actual_width", value.Device.ActualWidth); Comma(output);
            Property(output, "actual_height", value.Device.ActualHeight); Comma(output);
            Property(output, "observed_frame_rate", IsFinite(value.TimestampDeltaMedian) && value.TimestampDeltaMedian > 0.0 ? 1.0 / value.TimestampDeltaMedian : double.NaN); Comma(output);
            Property(output, "device_opened", value.Device.DeviceOpened); Comma(output);
            Property(output, "preview_observed", value.Device.PreviewObserved); Comma(output);
            Property(output, "provider_bound", value.Device.ProviderBound); Comma(output);
            Property(output, "provider_readiness_observed", value.Device.ProviderReadinessObserved); Comma(output);
            Property(output, "provider_frame_count", value.Device.ProviderFrameCount); Comma(output);
            Property(output, "bridge_observation_count", value.Device.BridgeObservationCount);
            output.Append("},");
            output.Append("\"capture_window\":");
            WriteCaptureWindow(output, value.CaptureWindow);
            output.Append(',');
            output.Append("\"source_convention\":{");
            Property(output, "id", value.SourceConventionId); Comma(output);
            Property(output, "axis_and_depth", value.AxisAndDepth); Comma(output);
            Property(output, "handedness", value.Handedness); Comma(output);
            Property(output, "mirror", value.Mirror);
            output.Append("},");
            output.Append("\"timestamp_summary\":{");
            Property(output, "strictly_increasing", value.TimestampsStrictlyIncreasing); Comma(output);
            Property(output, "delta_min", value.TimestampDeltaMin); Comma(output);
            Property(output, "delta_median", value.TimestampDeltaMedian); Comma(output);
            Property(output, "delta_max", value.TimestampDeltaMax); Comma(output);
            output.Append("\"timestamps\":[");
            for (int index = 0; index < value.Frames.Count; index++) { if (index > 0) Comma(output); NumberOrNull(output, value.Frames[index].TimestampSeconds); }
            output.Append("]},");
            output.Append("\"counts\":{");
            Property(output, "raw_frames", value.RawFrameCount); Comma(output);
            Property(output, "normalized_frames", value.NormalizedFrameCount); Comma(output);
            Property(output, "pose_present_frames", value.PosePresentFrameCount); Comma(output);
            Property(output, "left_hand_present_frames", value.LeftHandPresentFrameCount); Comma(output);
            Property(output, "right_hand_present_frames", value.RightHandPresentFrameCount); Comma(output);
            Property(output, "raw_pose_valid_landmarks", value.RawPoseValidLandmarkCount); Comma(output);
            Property(output, "raw_pose_landmark_capacity", value.RawFrameCount * 33); Comma(output);
            Property(output, "raw_pose_valid_ratio", Ratio(value.RawPoseValidLandmarkCount, value.RawFrameCount * 33)); Comma(output);
            Property(output, "raw_left_hand_valid_landmarks", value.RawLeftHandValidLandmarkCount); Comma(output);
            Property(output, "raw_left_hand_landmark_capacity", value.RawFrameCount * 21); Comma(output);
            Property(output, "raw_left_hand_valid_ratio", Ratio(value.RawLeftHandValidLandmarkCount, value.RawFrameCount * 21)); Comma(output);
            Property(output, "raw_right_hand_valid_landmarks", value.RawRightHandValidLandmarkCount); Comma(output);
            Property(output, "raw_right_hand_landmark_capacity", value.RawFrameCount * 21); Comma(output);
            Property(output, "raw_right_hand_valid_ratio", Ratio(value.RawRightHandValidLandmarkCount, value.RawFrameCount * 21)); Comma(output);
            Property(output, "raw_valid_landmarks", value.RawPoseValidLandmarkCount + value.RawLeftHandValidLandmarkCount + value.RawRightHandValidLandmarkCount); Comma(output);
            Property(output, "raw_landmark_capacity", value.RawFrameCount * 75); Comma(output);
            Property(output, "raw_valid_ratio", Ratio(value.RawPoseValidLandmarkCount + value.RawLeftHandValidLandmarkCount + value.RawRightHandValidLandmarkCount, value.RawFrameCount * 75)); Comma(output);
            Property(output, "normalized_valid_landmarks", value.NormalizedValidLandmarkCount); Comma(output);
            Property(output, "normalized_landmark_capacity", value.NormalizedFrameCount * 16); Comma(output);
            Property(output, "normalized_valid_ratio", Ratio(value.NormalizedValidLandmarkCount, value.NormalizedFrameCount * 16)); Comma(output);
            Property(output, "interpolated_landmarks", value.InterpolatedLandmarkCount); Comma(output);
            Property(output, "timestamp_boundaries", value.TimestampBoundaryCount); Comma(output);
            Property(output, "reappearances", value.ReappearanceCount); Comma(output);
            Property(output, "strong_depth_boundaries", value.StrongDepthBoundaryCount);
            output.Append("},");
            output.Append("\"body_transform\":{");
            Property(output, "available_frames", value.BodyTransformAvailableFrameCount); Comma(output);
            Property(output, "available_ratio", Ratio(value.BodyTransformAvailableFrameCount, value.RawFrameCount)); Comma(output);
            Property(output, "shoulder_width_raw_min", value.ShoulderWidthMin); Comma(output);
            Property(output, "shoulder_width_raw_median", value.ShoulderWidthMedian); Comma(output);
            Property(output, "shoulder_width_raw_max", value.ShoulderWidthMax);
            output.Append("},");
            Property(output, "finalization_state", value.FinalizationState); Comma(output);
            Property(output, "pre_export_objectively_valid", value.PreExportObjectivelyValid); Comma(output);
            output.Append("\"objective_invalid_reasons\":[");
            for (int index = 0; index < value.ObjectiveInvalidReasons.Count; index++) { if (index > 0) Comma(output); String(output, value.ObjectiveInvalidReasons[index]); }
            output.Append("],");
            output.Append("\"recognition\":");
            WriteRecognition(output, value.Recognition);
            output.Append(',');
            output.Append("\"failure\":");
            if (value.Failure == null) output.Append("null");
            else
            {
                output.Append('{'); Property(output, "exception_type", value.Failure.ExceptionType); Comma(output); Property(output, "message", value.Failure.Message); output.Append('}');
            }
            output.Append(',');
            output.Append("\"hand_identity_diagnostics\":{");
            Property(output, "duplicate_frames", value.DuplicateHandednessFrameCount); Comma(output);
            Property(output, "resolved_by_pose_wrist", value.ResolvedDuplicateHandednessFrameCount); Comma(output);
            Property(output, "ambiguous", value.AmbiguousDuplicateHandednessFrameCount); Comma(output);
            Property(output, "provider_frame_ratio", value.DuplicateHandednessFrameRatio); Comma(output);
            output.Append("\"events\":[");
            for (int index = 0; index < value.HandIdentityDiagnostics.Count; index++)
            {
                if (index > 0) Comma(output);
                WriteHandIdentityDiagnostic(output, value.HandIdentityDiagnostics[index]);
            }
            output.Append("]},");
            output.Append("\"frames\":[");
            for (int frameIndex = 0; frameIndex < value.Frames.Count; frameIndex++)
            {
                if (frameIndex > 0) Comma(output);
                WriteFrame(output, value.Frames[frameIndex], frameIndex);
            }
            output.Append(']');
            output.Append('}');
            output.Append('\n');
            return output.ToString();
        }

        static void WriteRecognition(StringBuilder output, PhysicalRecognitionEvidence value)
        {
            if (value == null) { output.Append("null"); return; }
            output.Append('{');
            Property(output, "predicted_word", value.PredictedWord); Comma(output);
            Property(output, "winning_prototype_id", value.WinningPrototypeId); Comma(output);
            output.Append("\"word_minima\":{");
            Property(output, "GREETING", value.GreetingDistance); Comma(output);
            Property(output, "LOVE", value.LoveDistance); Comma(output);
            Property(output, "THANK_YOU", value.ThankYouDistance);
            output.Append("},");
            Property(output, "best_distance", value.BestDistance); Comma(output);
            Property(output, "frozen_threshold", value.Threshold); Comma(output);
            Property(output, "threshold_rule", Step9CPhysicalCompatibilityConstants.ThresholdRule); Comma(output);
            Property(output, "accepted", value.Accepted); Comma(output);
            Property(output, "expected_word_match", value.ExpectedWordMatch); Comma(output);
            Property(output, "accepted_and_correct", value.AcceptedAndCorrect); Comma(output);
            output.Append("\"template_distances\":[");
            for (int index = 0; index < value.TemplateDistances.Count; index++)
            {
                if (index > 0) Comma(output);
                PhysicalTemplateDistanceEvidence distance = value.TemplateDistances[index];
                output.Append('{');
                Property(output, "prototype_id", distance.PrototypeId); Comma(output);
                Property(output, "word", distance.PrototypeWord); Comma(output);
                Property(output, "speed", distance.PrototypeSpeed); Comma(output);
                Property(output, "distance", distance.Distance); Comma(output);
                Property(output, "path_length", distance.PathLength);
                output.Append('}');
            }
            output.Append("]}");
        }

        static void WriteCaptureWindow(StringBuilder output, PhysicalCaptureWindowSnapshot value)
        {
            if (value == null) { output.Append("null"); return; }
            output.Append('{');
            Property(output, "prepare_after_lifetime_provider_frame_index", value.PrepareAfterLifetimeProviderFrameIndex); Comma(output);
            Property(output, "prepare_after_provider_timestamp_seconds", value.PrepareAfterProviderTimestampSeconds); Comma(output);
            Property(output, "start_next_lifetime_provider_frame_index", value.StartNextLifetimeProviderFrameIndex); Comma(output);
            Property(output, "start_previous_provider_timestamp_seconds", value.StartPreviousProviderTimestampSeconds); Comma(output);
            Property(output, "first_capture_provider_frame_index", value.FirstCaptureProviderFrameIndex); Comma(output);
            Property(output, "first_capture_lifetime_provider_frame_index", value.FirstCaptureLifetimeProviderFrameIndex); Comma(output);
            Property(output, "first_capture_provider_timestamp_seconds", value.FirstCaptureProviderTimestampSeconds); Comma(output);
            Property(output, "final_capture_provider_frame_index", value.FinalCaptureProviderFrameIndex); Comma(output);
            Property(output, "final_capture_lifetime_provider_frame_index", value.FinalCaptureLifetimeProviderFrameIndex); Comma(output);
            Property(output, "final_capture_provider_timestamp_seconds", value.FinalCaptureProviderTimestampSeconds); Comma(output);
            Property(output, "provider_frame_count", value.ProviderFrameCount); Comma(output);
            Property(output, "bridge_observation_count", value.BridgeObservationCount);
            output.Append('}');
        }

        static void WriteFrame(StringBuilder output, PhysicalFrameEvidence frame, int frameIndex)
        {
            bool[] mask = frame.CopyNormalizedMask();
            double[] coordinates = frame.CopyNormalizedCoordinates();
            output.Append('{');
            Property(output, "frame_index", frameIndex); Comma(output);
            Property(output, "timestamp_seconds", frame.TimestampSeconds); Comma(output);
            Property(output, "pose_present", frame.PosePresent); Comma(output);
            Property(output, "left_hand_present", frame.LeftHandPresent); Comma(output);
            Property(output, "right_hand_present", frame.RightHandPresent); Comma(output);
            Property(output, "pose_valid_landmarks", frame.PoseValidLandmarks); Comma(output);
            Property(output, "left_hand_valid_landmarks", frame.LeftHandValidLandmarks); Comma(output);
            Property(output, "right_hand_valid_landmarks", frame.RightHandValidLandmarks); Comma(output);
            Property(output, "body_transform_available", frame.BodyTransformAvailable); Comma(output);
            Property(output, "shoulder_width_raw", frame.ShoulderWidthRaw); Comma(output);
            Property(output, "has_normalized_frame", frame.HasNormalizedFrame); Comma(output);
            output.Append("\"normalized_mask\":[");
            for (int landmark = 0; landmark < 16; landmark++) { if (landmark > 0) Comma(output); output.Append(mask[landmark] ? "true" : "false"); }
            output.Append("],\"normalized_coordinates\":[");
            for (int landmark = 0; landmark < 16; landmark++)
            {
                if (landmark > 0) Comma(output);
                output.Append('[');
                for (int axis = 0; axis < 3; axis++)
                {
                    if (axis > 0) Comma(output);
                    NumberOrNull(output, mask[landmark] ? coordinates[landmark * 3 + axis] : double.NaN);
                }
                output.Append(']');
            }
            output.Append("]}");
        }

        static void WriteHandIdentityDiagnostic(StringBuilder output, Step9CHandIdentityDiagnostic value)
        {
            output.Append('{');
            Property(output, "provider_frame_index", value.ProviderFrameIndex); Comma(output);
            Property(output, "provider_timestamp_seconds", value.ProviderTimestampSeconds); Comma(output);
            Property(output, "hand_frame_timestamp_seconds", value.HandFrameTimestampSeconds); Comma(output);
            Property(output, "source_hand_count", value.SourceHandCount); Comma(output);
            output.Append("\"source_hands\":[{");
            Property(output, "category", value.SourceCategory0); Comma(output); Property(output, "score", value.SourceScore0); Comma(output);
            Property(output, "wrist_x", value.Hand0WristX); Comma(output); Property(output, "wrist_y", value.Hand0WristY);
            output.Append("},{");
            Property(output, "category", value.SourceCategory1); Comma(output); Property(output, "score", value.SourceScore1); Comma(output);
            Property(output, "wrist_x", value.Hand1WristX); Comma(output); Property(output, "wrist_y", value.Hand1WristY);
            output.Append("}],\"pose_wrists\":{");
            Property(output, "left_x", value.PoseLeftWristX); Comma(output); Property(output, "left_y", value.PoseLeftWristY); Comma(output);
            Property(output, "left_visibility", value.PoseLeftWristVisibility); Comma(output);
            Property(output, "right_x", value.PoseRightWristX); Comma(output); Property(output, "right_y", value.PoseRightWristY); Comma(output);
            Property(output, "right_visibility", value.PoseRightWristVisibility);
            output.Append("},\"assignment\":{");
            Property(output, "direct_cost", value.DirectAssignmentCost); Comma(output);
            Property(output, "swapped_cost", value.SwappedAssignmentCost); Comma(output);
            Property(output, "best_max_distance", value.BestAssignmentMaxDistance); Comma(output);
            Property(output, "margin", value.AssignmentMargin); Comma(output);
            Property(output, "pose_wrist_separation", value.PoseWristSeparation); Comma(output);
            Property(output, "selected_hand0_identity", value.SelectedHand0Identity);
            output.Append("},");
            Property(output, "outcome", value.Outcome.ToString()); Comma(output);
            Property(output, "reason", value.Reason);
            output.Append('}');
        }

        static void Property(StringBuilder output, string name, string value) { String(output, name); output.Append(':'); String(output, value); }
        static void Property(StringBuilder output, string name, int value) { String(output, name); output.Append(':').Append(value.ToString(CultureInfo.InvariantCulture)); }
        static void Property(StringBuilder output, string name, long value) { String(output, name); output.Append(':').Append(value.ToString(CultureInfo.InvariantCulture)); }
        static void Property(StringBuilder output, string name, bool value) { String(output, name); output.Append(':').Append(value ? "true" : "false"); }
        static void Property(StringBuilder output, string name, double value) { String(output, name); output.Append(':'); NumberOrNull(output, value); }
        static void NumberOrNull(StringBuilder output, double value) => output.Append(IsFinite(value) ? value.ToString("R", CultureInfo.InvariantCulture) : "null");
        static double Ratio(int numerator, int denominator) => denominator == 0 ? double.NaN : (double)numerator / denominator;
        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        static bool IsCategory(string value) => string.Equals(value, "Left", StringComparison.Ordinal) || string.Equals(value, "Right", StringComparison.Ordinal);
        static bool IsScore(double value) => IsFinite(value) && value >= 0.0 && value <= 1.0;
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
                            if (character < 0x20) output.Append("\\u").Append(((int)character).ToString("x4"));
                            else output.Append(character);
                            break;
                    }
                }
            }
            output.Append('"');
        }
    }

    public sealed class PhysicalEvidenceExportReceipt
    {
        public string EvidencePath { get; }
        public string ChecksumPath { get; }
        public string Sha256 { get; }
        public long SizeBytes { get; }
        public bool HashVerified { get; }

        internal PhysicalEvidenceExportReceipt(string evidencePath, string checksumPath, string sha256, long sizeBytes, bool hashVerified)
        {
            EvidencePath = evidencePath;
            ChecksumPath = checksumPath;
            Sha256 = sha256;
            SizeBytes = sizeBytes;
            HashVerified = hashVerified;
        }
    }

    public static class PhysicalCaptureEvidenceFile
    {
        static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(false, true);

        public static PhysicalEvidenceExportReceipt Export(string path, PhysicalCaptureEvidence evidence)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("evidence path is required", nameof(path));
            string fullPath = Path.GetFullPath(path);
            string checksumPath = fullPath + ".sha256";
            if (File.Exists(fullPath) || File.Exists(checksumPath)) throw new IOException("evidence export never overwrites an existing capture");
            string directory = Path.GetDirectoryName(fullPath);
            if (string.IsNullOrEmpty(directory)) throw new InvalidOperationException("evidence directory is invalid");
            Directory.CreateDirectory(directory);

            byte[] bytes = Utf8NoBom.GetBytes(PhysicalCaptureEvidenceJson.Serialize(evidence));
            string temporaryPath = fullPath + ".tmp";
            File.WriteAllBytes(temporaryPath, bytes);
            File.Move(temporaryPath, fullPath);
            string hash = Sha256(bytes);
            File.WriteAllText(checksumPath, hash + "  " + Path.GetFileName(fullPath) + "\n", Utf8NoBom);
            bool verified = Verify(fullPath, checksumPath);
            return new PhysicalEvidenceExportReceipt(fullPath, checksumPath, hash, bytes.LongLength, verified);
        }

        public static bool Verify(string evidencePath, string checksumPath)
        {
            try
            {
                if (!File.Exists(evidencePath) || !File.Exists(checksumPath)) return false;
                string expectedLine = File.ReadAllText(checksumPath, Utf8NoBom);
                string fileName = Path.GetFileName(evidencePath);
                if (expectedLine.Length != 64 + 2 + fileName.Length + 1 || expectedLine[64] != ' ' || expectedLine[65] != ' ' || expectedLine[expectedLine.Length - 1] != '\n') return false;
                string expectedHash = expectedLine.Substring(0, 64);
                if (expectedHash.Any(character => !(character >= '0' && character <= '9') && !(character >= 'a' && character <= 'f'))) return false;
                if (!string.Equals(expectedLine.Substring(66, fileName.Length), fileName, StringComparison.Ordinal)) return false;
                return string.Equals(expectedHash, Sha256(File.ReadAllBytes(evidencePath)), StringComparison.Ordinal);
            }
            catch
            {
                return false;
            }
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
    }
}
