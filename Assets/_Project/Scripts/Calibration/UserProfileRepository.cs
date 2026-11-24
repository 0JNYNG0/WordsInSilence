using System;
using System.IO;
using UnityEngine;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// UserCalibrationProfile을 persistentDataPath/Profiles/ 에 저장/로드한다.
    /// 프로필 ID = 파일명 (확장자 없이).
    /// </summary>
    public static class UserProfileRepository
    {
        static string ProfilesDir => Path.Combine(Application.persistentDataPath, "Profiles");

        public static void Save(UserCalibrationProfile profile)
        {
            if (profile == null) throw new ArgumentNullException(nameof(profile));
            if (string.IsNullOrEmpty(profile.UserProfileId))
                throw new ArgumentException("UserProfileId is required.");

            profile.UpdatedUtc = DateTime.UtcNow.ToString("o");
            if (string.IsNullOrEmpty(profile.CreatedUtc))
                profile.CreatedUtc = profile.UpdatedUtc;

            Directory.CreateDirectory(ProfilesDir);
            string path = Path.Combine(ProfilesDir, $"{profile.UserProfileId}.json");
            string json = JsonUtility.ToJson(profile, prettyPrint: true);
            File.WriteAllText(path, json);
        }

        public static UserCalibrationProfile Load(string profileId)
        {
            if (string.IsNullOrEmpty(profileId)) return null;
            string path = Path.Combine(ProfilesDir, $"{profileId}.json");
            if (!File.Exists(path)) return null;
            string json = File.ReadAllText(path);
            return JsonUtility.FromJson<UserCalibrationProfile>(json);
        }

        public static void Delete(string profileId)
        {
            if (string.IsNullOrEmpty(profileId)) return;
            string path = Path.Combine(ProfilesDir, $"{profileId}.json");
            if (File.Exists(path)) File.Delete(path);
        }

        public static string[] ListAll()
        {
            if (!Directory.Exists(ProfilesDir)) return Array.Empty<string>();
            string[] files = Directory.GetFiles(ProfilesDir, "*.json");
            var names = new string[files.Length];
            for (int i = 0; i < files.Length; i++)
                names[i] = Path.GetFileNameWithoutExtension(files[i]);
            return names;
        }

        /// <summary>
        /// 카메라 장치/해상도/회전이 변경되었는지 확인한다.
        /// </summary>
        public static bool HasCameraChanged(UserCalibrationProfile profile, CameraFingerprint current)
        {
            if (profile == null || current == null) return true;
            var saved = profile.Camera;
            if (saved == null) return true;
            return saved.DeviceName != current.DeviceName
                || saved.Width != current.Width
                || saved.Height != current.Height
                || saved.RotationDegrees != current.RotationDegrees;
        }

        /// <summary>
        /// 현재 어깨너비가 저장된 기준과 허용 범위 내인지 확인한다.
        /// </summary>
        public static bool IsBodyScaleConsistent(UserCalibrationProfile profile, float currentShoulderWidth, float toleranceRatio = 0.25f)
        {
            if (profile == null) return false;
            float saved = profile.BodyScale.ShoulderWidthNorm;
            if (saved < 0.01f) return false;
            float diff = Mathf.Abs(currentShoulderWidth - saved) / saved;
            return diff <= toleranceRatio;
        }
    }
}
