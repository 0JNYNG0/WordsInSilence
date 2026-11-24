using System;
using System.IO;
using UnityEngine;

namespace WordsInSilence.Personalization
{
    /// <summary>
    /// 개인 등록 데이터를 persistentDataPath/PersonalData/{profileId}/ 에 저장한다.
    /// 원본 영상은 저장하지 않는다.
    /// </summary>
    public static class PersonalDataRepository
    {
        static string BaseDir => Path.Combine(Application.persistentDataPath, "PersonalData");

        static string ProfileDir(string profileId) =>
            Path.Combine(BaseDir, profileId);

        static string EnrollmentPath(string profileId, string signId) =>
            Path.Combine(ProfileDir(profileId), $"enrollment_{signId}.json");

        // ─── Enrollment ────────────────────────────────────────────────────

        public static void SaveEnrollment(string profileId, PersonalSignEnrollment enrollment)
        {
            if (string.IsNullOrEmpty(profileId) || enrollment == null) return;

            string dir = ProfileDir(profileId);
            Directory.CreateDirectory(dir);

            string path = EnrollmentPath(profileId, enrollment.SignId);
            string json = JsonUtility.ToJson(enrollment, prettyPrint: true);

            // Atomic write: 임시 파일에 쓰고 이동
            string tmpPath = path + ".tmp";
            try
            {
                File.WriteAllText(tmpPath, json);
                if (File.Exists(path))
                {
                    // 백업
                    string bakPath = path + ".bak";
                    if (File.Exists(bakPath)) File.Delete(bakPath);
                    File.Move(path, bakPath);
                }
                File.Move(tmpPath, path);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[PersonalDataRepository] Save failed: {e.Message}");
                // 임시 파일 정리
                if (File.Exists(tmpPath)) File.Delete(tmpPath);
            }
        }

        public static PersonalSignEnrollment LoadEnrollment(string profileId, string signId)
        {
            string path = EnrollmentPath(profileId, signId);

            // 주 파일 시도
            if (File.Exists(path))
            {
                try
                {
                    string json = File.ReadAllText(path);
                    return JsonUtility.FromJson<PersonalSignEnrollment>(json);
                }
                catch
                {
                    // 손상 → 백업에서 복구 시도
                    Debug.LogWarning($"[PersonalDataRepository] Corrupt file, trying backup: {path}");
                }
            }

            // 백업 파일 시도
            string bakPath = path + ".bak";
            if (File.Exists(bakPath))
            {
                try
                {
                    string json = File.ReadAllText(bakPath);
                    var result = JsonUtility.FromJson<PersonalSignEnrollment>(json);
                    // 복구 성공 → 주 파일 복원
                    File.Copy(bakPath, path, overwrite: true);
                    return result;
                }
                catch
                {
                    Debug.LogWarning($"[PersonalDataRepository] Backup also corrupt: {bakPath}");
                }
            }

            return null;
        }

        public static void DeleteEnrollment(string profileId, string signId)
        {
            string path = EnrollmentPath(profileId, signId);
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(path + ".bak")) File.Delete(path + ".bak");
            if (File.Exists(path + ".tmp")) File.Delete(path + ".tmp");
        }

        public static void DeleteAllForProfile(string profileId)
        {
            string dir = ProfileDir(profileId);
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }

        /// <summary>
        /// 프로필 폴더의 모든 등록된 signId를 반환한다.
        /// </summary>
        public static string[] ListEnrolledSigns(string profileId)
        {
            string dir = ProfileDir(profileId);
            if (!Directory.Exists(dir)) return Array.Empty<string>();

            string[] files = Directory.GetFiles(dir, "enrollment_*.json");
            var signs = new string[files.Length];
            for (int i = 0; i < files.Length; i++)
            {
                string name = Path.GetFileNameWithoutExtension(files[i]);
                // "enrollment_SIGN002" → "SIGN002"
                signs[i] = name.Length > 11 ? name.Substring(11) : name;
            }
            return signs;
        }
    }
}
