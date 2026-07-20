using UnityEngine;
using WordsInSilence.Core;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// 정규화된 손 랜드마크를 SignDictionary 템플릿과 비교해 포즈를 인식한다.
    /// 정규화: wrist(0번)를 원점으로 이동, wrist-MCP9(9번) 거리를 스케일로 나눔.
    /// </summary>
    public class PoseTemplateEvaluator
    {
        readonly SignDictionary _dict;

        public PoseTemplateEvaluator(SignDictionary dict)
        {
            _dict = dict;
        }

        /// <summary>
        /// SignMotionSample에서 평균 포즈를 계산한 뒤 템플릿과 매칭한다.
        /// </summary>
        public SignEntry? Evaluate(SignMotionSample sample)
        {
            if (sample == null || sample.Frames == null || sample.Frames.Length == 0)
                return null;

            var avgPose = ComputeAveragePose(sample);
            if (avgPose == null) return null;

            return Evaluate(avgPose);
        }

        /// <summary>
        /// 21개 원시 랜드마크를 정규화한 뒤 템플릿과 매칭한다.
        /// </summary>
        public SignEntry? Evaluate(Vector3[] rawLandmarks)
        {
            if (rawLandmarks == null || rawLandmarks.Length != 21) return null;
            if (_dict == null || _dict.signs == null || _dict.signs.Count == 0) return null;

            var norm = Normalize(rawLandmarks);

            float bestDist = float.MaxValue;
            SignEntry? bestEntry = null;

            foreach (var entry in _dict.signs)
            {
                if (entry.landmarks == null || entry.landmarks.Length != 21) continue;

                var normTemplate = Normalize(entry.landmarks);
                float dist = AverageDistance(norm, normTemplate);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestEntry = entry;
                }
            }

            if (bestEntry == null) return null;

            Debug.Log($"[PoseTemplateEvaluator] 최근접 포즈: {bestEntry.Value.signId}, 거리={bestDist:F4}, 임계값={bestEntry.Value.threshold}");

            if (bestDist > bestEntry.Value.threshold) return null;

            return bestEntry;
        }

        // ─── Static helpers ──────────────────────────────────────────────────

        /// <summary>
        /// SignMotionSample의 유효 프레임들에서 첫 번째 손 좌표를 추출해 성분별 평균을 반환한다.
        /// 반환값은 정규화 전 원시 좌표 (Vector3[21]).
        /// </summary>
        public static Vector3[] ComputeAveragePose(SignMotionSample sample)
        {
            var accum = new Vector3[21];
            int count = 0;

            foreach (var frame in sample.Frames)
            {
                if (frame.Hands == null || frame.Hands.Length == 0) continue;

                var hand = frame.Hands[0];
                if (hand.Points == null || hand.Points.Length != 21) continue;

                for (int i = 0; i < 21; i++)
                    accum[i] += new Vector3(hand.Points[i].X, hand.Points[i].Y, hand.Points[i].Z);

                count++;
            }

            if (count == 0) return null;

            for (int i = 0; i < 21; i++)
                accum[i] /= count;

            return accum;
        }

        static Vector3[] Normalize(Vector3[] raw)
        {
            var pts = new Vector3[21];
            // Z는 MediaPipe 추정 깊이라 노이즈가 크므로 2D(X,Y)만 사용
            for (int i = 0; i < 21; i++)
                pts[i] = new Vector3(raw[i].x, raw[i].y, 0f);

            // 1. Wrist (0번)를 원점으로
            var origin = pts[0];
            for (int i = 0; i < 21; i++)
                pts[i] -= origin;

            // 2. wrist-MCP9 (middle finger MCP) 2D 거리를 스케일로 나눔
            float scale = pts[9].magnitude;
            if (scale < 1e-6f) return pts;

            for (int i = 0; i < 21; i++)
                pts[i] /= scale;

            return pts;
        }

        static float AverageDistance(Vector3[] a, Vector3[] b)
        {
            float sum = 0f;
            for (int i = 0; i < 21; i++)
                sum += Vector3.Distance(a[i], b[i]);
            return sum / 21f;
        }
    }
}
