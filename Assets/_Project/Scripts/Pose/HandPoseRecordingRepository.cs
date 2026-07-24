using System.IO;
using UnityEngine;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// HandPoseRecording을 JSON 파일로 저장/로드한다.
    /// 레퍼런스: Application.persistentDataPath/Recordings/{motionId}.json
    /// 학습샘플: Application.persistentDataPath/TrainingSamples/{label}_{timestamp}.json
    /// </summary>
    public static class HandPoseRecordingRepository
    {
        static string SaveDir => Path.Combine(Application.persistentDataPath, "Recordings");
        static string TrainingSamplesDir => Path.Combine(Application.persistentDataPath, "TrainingSamples");

        public static void Save(HandPoseRecording recording)
        {
            Directory.CreateDirectory(SaveDir);
            string path = Path.Combine(SaveDir, $"{recording.MotionId}.json");
            string json = JsonUtility.ToJson(recording, prettyPrint: true);
            File.WriteAllText(path, json);
            Debug.Log($"[HandPoseRecordingRepository] 저장: {path}");
        }

        public static HandPoseRecording Load(string motionId)
        {
            string path = Path.Combine(SaveDir, $"{motionId}.json");
            if (!File.Exists(path)) return null;
            string json = File.ReadAllText(path);
            return JsonUtility.FromJson<HandPoseRecording>(json);
        }

        public static string[] ListAll()
        {
            if (!Directory.Exists(SaveDir)) return System.Array.Empty<string>();
            string[] files = Directory.GetFiles(SaveDir, "*.json");
            var names = new string[files.Length];
            for (int i = 0; i < files.Length; i++)
                names[i] = Path.GetFileNameWithoutExtension(files[i]);
            return names;
        }

        // ─── 학습 샘플 (TrainingSamples/) ────────────────────────────────────

        /// <summary>
        /// 학습 샘플을 TrainingSamples/{DisplayName}_{timestamp}.json 으로 저장한다.
        /// DisplayName이 레이블로 사용된다.
        /// </summary>
        public static void SaveTrainingSample(HandPoseRecording recording)
        {
            Directory.CreateDirectory(TrainingSamplesDir);
            string label = string.IsNullOrEmpty(recording.DisplayName)
                           ? recording.MotionId
                           : recording.DisplayName;
            string ts   = System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString();
            string path = Path.Combine(TrainingSamplesDir, $"{label}_{ts}.json");
            string json = JsonUtility.ToJson(recording, prettyPrint: true);
            File.WriteAllText(path, json);
            Debug.Log($"[HandPoseRecordingRepository] 샘플 저장: {path}");
        }

        /// <summary>TrainingSamples/ 폴더의 모든 JSON 절대 경로 목록을 반환한다.</summary>
        public static string[] ListTrainingSamples()
        {
            if (!Directory.Exists(TrainingSamplesDir)) return System.Array.Empty<string>();
            return Directory.GetFiles(TrainingSamplesDir, "*.json");
        }

        /// <summary>TrainingSamples/ 폴더의 경로를 반환한다 (Python에 넘겨주기 용).</summary>
        public static string GetTrainingSamplesPath() => TrainingSamplesDir;
    }
}
