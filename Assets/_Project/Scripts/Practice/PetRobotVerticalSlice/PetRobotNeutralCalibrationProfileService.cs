using System;
using System.IO;
using UnityEngine;
using WordsInSilence.Calibration;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum NeutralCalibrationProfileLoadStatus
    {
        Valid,
        Missing,
        Corrupt,
        Invalid,
        CameraMismatch,
    }

    public enum NeutralCalibrationProfileSaveStatus
    {
        Saved,
        ExistingProfilePreserved,
        ValidationFailed,
        TransactionFailed,
    }

    public sealed class NeutralCalibrationProfileLoadResult
    {
        public NeutralCalibrationProfileLoadStatus Status { get; }
        public UserCalibrationProfile Profile { get; }
        public string Message { get; }
        public bool NeedsCalibration => Status != NeutralCalibrationProfileLoadStatus.Valid;

        public NeutralCalibrationProfileLoadResult(
            NeutralCalibrationProfileLoadStatus status,
            UserCalibrationProfile profile,
            string message)
        {
            Status = status;
            Profile = profile;
            Message = message ?? string.Empty;
        }
    }

    public sealed class NeutralCalibrationProfileSaveResult
    {
        public NeutralCalibrationProfileSaveStatus Status { get; }
        public string ProfilePath { get; }
        public string Message { get; }
        public bool IsSaved => Status == NeutralCalibrationProfileSaveStatus.Saved;

        public NeutralCalibrationProfileSaveResult(
            NeutralCalibrationProfileSaveStatus status,
            string profilePath,
            string message)
        {
            Status = status;
            ProfilePath = profilePath ?? string.Empty;
            Message = message ?? string.Empty;
        }
    }

    /// <summary>Injectable filesystem boundary used by the same production transaction in tests.</summary>
    public interface INeutralCalibrationProfileFileSystem
    {
        bool FileExists(string path);
        void CreateDirectory(string path);
        string ReadAllText(string path);
        void WriteAllText(string path, string contents);
        void MoveFile(string source, string destination);
        void ReplaceFile(string source, string destination, string backup);
        void DeleteFile(string path);
    }

    public sealed class SystemNeutralCalibrationProfileFileSystem : INeutralCalibrationProfileFileSystem
    {
        public bool FileExists(string path) => File.Exists(path);
        public void CreateDirectory(string path) => Directory.CreateDirectory(path);
        public string ReadAllText(string path) => File.ReadAllText(path);
        public void WriteAllText(string path, string contents) => File.WriteAllText(path, contents);
        public void MoveFile(string source, string destination) => File.Move(source, destination);
        public void ReplaceFile(string source, string destination, string backup) =>
            File.Replace(source, destination, backup);
        public void DeleteFile(string path)
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    /// <summary>
    /// Current-native, learner-keyed neutral profile persistence. It reuses only the data schema;
    /// it never calls the legacy calibration controller or the direct-overwrite repository save.
    /// </summary>
    public sealed class PetRobotNeutralCalibrationProfileService
    {
        public const int SupportedSchemaVersion = 1;
        public const float BodyScaleToleranceRatio = 0.25f;

        readonly string learnerProfileId;
        readonly string profilesRoot;
        readonly INeutralCalibrationProfileFileSystem fileSystem;
        readonly Func<DateTime> utcNow;

        public string LearnerProfileId => learnerProfileId;
        public string ProfilePath => Path.Combine(profilesRoot, learnerProfileId + ".json");

        public PetRobotNeutralCalibrationProfileService(
            string learnerProfileId,
            string profilesRoot = null,
            INeutralCalibrationProfileFileSystem fileSystem = null,
            Func<DateTime> utcNow = null)
        {
            if (!IsSafeProfileId(learnerProfileId))
                throw new ArgumentException("learner profile ID must contain only letters, digits, dot, dash, or underscore", nameof(learnerProfileId));
            this.learnerProfileId = learnerProfileId;
            this.profilesRoot = profilesRoot ?? Path.Combine(Application.persistentDataPath, "Profiles");
            this.fileSystem = fileSystem ?? new SystemNeutralCalibrationProfileFileSystem();
            this.utcNow = utcNow ?? (() => DateTime.UtcNow);
        }

        public NeutralCalibrationProfileLoadResult Load(CameraFingerprint currentCamera)
        {
            if (!fileSystem.FileExists(ProfilePath))
                return new NeutralCalibrationProfileLoadResult(
                    NeutralCalibrationProfileLoadStatus.Missing, null, "learner neutral profile is missing");

            UserCalibrationProfile profile;
            try
            {
                profile = Deserialize(fileSystem.ReadAllText(ProfilePath));
            }
            catch (Exception exception)
            {
                return new NeutralCalibrationProfileLoadResult(
                    NeutralCalibrationProfileLoadStatus.Corrupt, null,
                    "learner neutral profile could not be parsed: " + exception.Message);
            }

            string validationFailure;
            if (!ValidateProfile(profile, learnerProfileId, out validationFailure))
                return new NeutralCalibrationProfileLoadResult(
                    NeutralCalibrationProfileLoadStatus.Invalid, null, validationFailure);
            if (!CameraMatches(profile.Camera, currentCamera))
                return new NeutralCalibrationProfileLoadResult(
                    NeutralCalibrationProfileLoadStatus.CameraMismatch, null,
                    "current camera fingerprint does not match the saved neutral profile");
            return new NeutralCalibrationProfileLoadResult(
                NeutralCalibrationProfileLoadStatus.Valid, profile, string.Empty);
        }

        public NeutralCalibrationProfileSaveResult Save(
            UserCalibrationProfile profile,
            bool allowExplicitRecalibrationUpdate)
        {
            string validationFailure;
            if (!ValidateProfile(profile, learnerProfileId, out validationFailure))
                return new NeutralCalibrationProfileSaveResult(
                    NeutralCalibrationProfileSaveStatus.ValidationFailed, ProfilePath, validationFailure);

            bool existing = fileSystem.FileExists(ProfilePath);
            if (existing && !allowExplicitRecalibrationUpdate)
                return new NeutralCalibrationProfileSaveResult(
                    NeutralCalibrationProfileSaveStatus.ExistingProfilePreserved,
                    ProfilePath,
                    "an existing valid profile was preserved; explicit recalibration update was not authorized");

            string stamp = utcNow().ToUniversalTime().ToString("o");
            profile.UpdatedUtc = stamp;
            if (string.IsNullOrWhiteSpace(profile.CreatedUtc)) profile.CreatedUtc = stamp;

            string transactionId = Guid.NewGuid().ToString("N");
            string temporaryPath = ProfilePath + "." + transactionId + ".tmp";
            string backupPath = ProfilePath + "." + transactionId + ".bak";
            bool committedNewFile = false;
            bool replacedExisting = false;
            try
            {
                fileSystem.CreateDirectory(profilesRoot);
                fileSystem.WriteAllText(temporaryPath, JsonUtility.ToJson(profile, true));
                UserCalibrationProfile temporaryProfile = Deserialize(fileSystem.ReadAllText(temporaryPath));
                if (!ValidateProfile(temporaryProfile, learnerProfileId, out validationFailure))
                    throw new InvalidDataException("temporary profile validation failed: " + validationFailure);

                if (existing)
                {
                    fileSystem.ReplaceFile(temporaryPath, ProfilePath, backupPath);
                    replacedExisting = true;
                }
                else
                {
                    fileSystem.MoveFile(temporaryPath, ProfilePath);
                    committedNewFile = true;
                }

                UserCalibrationProfile committed = Deserialize(fileSystem.ReadAllText(ProfilePath));
                if (!ValidateProfile(committed, learnerProfileId, out validationFailure))
                    throw new InvalidDataException("post-commit profile validation failed: " + validationFailure);
                if (!CameraMatches(committed.Camera, profile.Camera))
                    throw new InvalidDataException("post-commit camera fingerprint changed");

                fileSystem.DeleteFile(backupPath);
                return new NeutralCalibrationProfileSaveResult(
                    NeutralCalibrationProfileSaveStatus.Saved, ProfilePath, string.Empty);
            }
            catch (Exception exception)
            {
                try
                {
                    if (replacedExisting && fileSystem.FileExists(backupPath))
                    {
                        fileSystem.DeleteFile(ProfilePath);
                        fileSystem.MoveFile(backupPath, ProfilePath);
                    }
                    else if (committedNewFile)
                    {
                        fileSystem.DeleteFile(ProfilePath);
                    }
                }
                catch (Exception rollbackException)
                {
                    return new NeutralCalibrationProfileSaveResult(
                        NeutralCalibrationProfileSaveStatus.TransactionFailed,
                        ProfilePath,
                        exception.Message + "; rollback failed: " + rollbackException.Message);
                }
                finally
                {
                    SafeDelete(temporaryPath);
                    SafeDelete(backupPath);
                }

                return new NeutralCalibrationProfileSaveResult(
                    NeutralCalibrationProfileSaveStatus.TransactionFailed,
                    ProfilePath,
                    exception.GetType().Name + ": " + exception.Message);
            }
            finally
            {
                SafeDelete(temporaryPath);
                if (!replacedExisting || !fileSystem.FileExists(ProfilePath)) SafeDelete(backupPath);
            }
        }

        public static bool IsBodyScaleConsistent(UserCalibrationProfile profile, float currentShoulderWidth)
        {
            if (profile == null || profile.BodyScale == null || !IsFinite(currentShoulderWidth)) return false;
            float saved = profile.BodyScale.ShoulderWidthNorm;
            if (!IsFinite(saved) || saved < 0.01f) return false;
            return Mathf.Abs(currentShoulderWidth - saved) / saved <= BodyScaleToleranceRatio;
        }

        public static bool ValidateProfile(
            UserCalibrationProfile profile,
            string expectedLearnerProfileId,
            out string failure)
        {
            failure = string.Empty;
            if (profile == null) return FailValidation("profile is null", out failure);
            if (profile.SchemaVersion != SupportedSchemaVersion)
                return FailValidation("unsupported neutral profile schema", out failure);
            if (!string.Equals(profile.UserProfileId, expectedLearnerProfileId, StringComparison.Ordinal))
                return FailValidation("neutral profile learner ID mismatch", out failure);
            if (profile.BodyBaseline == null || profile.Camera == null || profile.BodyScale == null ||
                profile.Neutral == null || profile.SafeFrame == null || profile.Quality == null)
                return FailValidation("required neutral profile object is missing", out failure);
            if (!ValidateCamera(profile.Camera))
                return FailValidation("saved camera fingerprint is invalid", out failure);
            if (!IsFinite(profile.BodyScale.ShoulderWidthNorm) || profile.BodyScale.ShoulderWidthNorm < 0.01f)
                return FailValidation("saved shoulder width is invalid", out failure);
            if (!AllOptionalBodyScaleFinite(profile.BodyScale))
                return FailValidation("saved body-scale value is non-finite", out failure);
            if (!IsFinite(profile.Neutral.MotionNoiseFloor) || profile.Neutral.MotionNoiseFloor < 0f)
                return FailValidation("saved motion noise floor is invalid", out failure);
            if (profile.Neutral.HasArmAngleBaseline &&
                (!IsFinite(profile.Neutral.LeftArmAngleNeutral) || !IsFinite(profile.Neutral.RightArmAngleNeutral)))
                return FailValidation("saved arm-angle baseline is non-finite", out failure);
            if (profile.Neutral.HasArmVerticalBaseline &&
                (!IsFinite(profile.Neutral.LeftArmVerticalNeutral) || !IsFinite(profile.Neutral.RightArmVerticalNeutral)))
                return FailValidation("saved arm-vertical baseline is non-finite", out failure);
            if (profile.Neutral.HasWristBaseline &&
                (!IsFinite(profile.Neutral.LeftWristNeutralBR.x) || !IsFinite(profile.Neutral.LeftWristNeutralBR.y) ||
                 !IsFinite(profile.Neutral.RightWristNeutralBR.x) || !IsFinite(profile.Neutral.RightWristNeutralBR.y)))
                return FailValidation("saved wrist baseline is non-finite", out failure);
            if (profile.Neutral.HandsVisible && !profile.Neutral.HasWristBaseline)
                return FailValidation("hands-visible neutral requires a wrist baseline", out failure);
            if (!profile.Neutral.HandsVisible && !profile.Neutral.BottomExitAllowed)
                return FailValidation("hands-hidden neutral requires bottom-exit compatibility", out failure);
            if (!IsFinite(profile.Quality.ValidFrameRatio) || profile.Quality.ValidFrameRatio < 0f ||
                profile.Quality.ValidFrameRatio > 1f)
                return FailValidation("neutral profile quality ratio is invalid", out failure);
            if (!string.Equals(profile.Quality.Status, "good", StringComparison.Ordinal) &&
                !string.Equals(profile.Quality.Status, "marginal", StringComparison.Ordinal))
                return FailValidation("neutral profile quality status is not usable", out failure);
            return true;
        }

        public static bool CameraMatches(CameraFingerprint saved, CameraFingerprint current)
        {
            return ValidateCamera(saved) && ValidateCamera(current) &&
                string.Equals(saved.DeviceName, current.DeviceName, StringComparison.Ordinal) &&
                saved.Width == current.Width && saved.Height == current.Height &&
                saved.RotationDegrees == current.RotationDegrees &&
                saved.PreviewMirrored == current.PreviewMirrored &&
                saved.Fps.Equals(current.Fps);
        }

        static UserCalibrationProfile Deserialize(string json)
        {
            if (string.IsNullOrWhiteSpace(json)) throw new InvalidDataException("profile JSON is empty");
            UserCalibrationProfile profile = JsonUtility.FromJson<UserCalibrationProfile>(json);
            if (profile == null) throw new InvalidDataException("profile JSON returned null");
            return profile;
        }

        static bool ValidateCamera(CameraFingerprint value)
        {
            return value != null && !string.IsNullOrWhiteSpace(value.DeviceName) &&
                value.Width > 0 && value.Height > 0 && IsFinite(value.Fps) && value.Fps > 0f &&
                (value.RotationDegrees == 0 || value.RotationDegrees == 90 ||
                 value.RotationDegrees == 180 || value.RotationDegrees == 270);
        }

        static bool AllOptionalBodyScaleFinite(BodyScaleData value)
        {
            return (!value.HasUpperArmRatio ||
                    (IsFinite(value.LeftUpperArmOverShoulder) && IsFinite(value.RightUpperArmOverShoulder))) &&
                (!value.HasForearmRatio ||
                    (IsFinite(value.LeftForearmOverShoulder) && IsFinite(value.RightForearmOverShoulder))) &&
                (!value.HasTorsoRatio || IsFinite(value.TorsoOverShoulder));
        }

        static bool IsSafeProfileId(string value)
        {
            if (string.IsNullOrWhiteSpace(value) || value == "." || value == "..") return false;
            foreach (char character in value)
                if (!(char.IsLetterOrDigit(character) || character == '_' || character == '-' || character == '.'))
                    return false;
            return true;
        }

        static bool FailValidation(string message, out string failure)
        {
            failure = message;
            return false;
        }

        static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        void SafeDelete(string path)
        {
            try { fileSystem.DeleteFile(path); }
            catch (Exception) { }
        }
    }
}
