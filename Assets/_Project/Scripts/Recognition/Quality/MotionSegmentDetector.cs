using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Recognition.Canonical;

namespace WordsInSilence.Recognition.Quality
{
    /// <summary>
    /// CanonicalFrame[] 에서 동작 구간을 검출하고 품질을 평가한다.
    ///
    /// 기존 TrimInactive() (손 미검출 구간 단순 제거) 와는 독립된 별도 모듈이다.
    /// 기존 TrimInactive()는 legacy 경로에서 계속 사용한다.
    ///
    /// 활성 프레임 판정 기준:
    ///   - 손목 이동 속도 >= WristVelocityThreshold
    ///   - 또는 랜드마크 이동 에너지 >= LandmarkEnergyThreshold
    ///   손 검출 여부 자체는 속도 계산의 validity mask로만 사용하고
    ///   active 판정 조건에 포함하지 않는다.
    ///   → 정지한 손을 들고 있는 프레임은 활성으로 간주하지 않는다.
    ///
    /// 갭 처리: MaxInactiveGapFrames 이하의 짧은 비활성 구간은 동작 구간으로 연결한다.
    /// </summary>
    public static class MotionSegmentDetector
    {
        // ─── 결과 타입 ───────────────────────────────────────────────────────

        public sealed class SegmentResult
        {
            /// <summary>품질 상태.</summary>
            public FrameQualityStatus Status;

            /// <summary>Status == Analyzable 일 때의 동작 구간 프레임.</summary>
            public CanonicalFrame[] Segment;

            /// <summary>전체 프레임 중 손 검출 비율.</summary>
            public float HandFrameRatio;

            /// <summary>전체 프레임 중 pose 검출 비율.</summary>
            public float PoseFrameRatio;

            /// <summary>
            /// 양손 수어 여부 추정.
            /// 구간 검출에 성공한 경우 해당 구간 기준, 실패하면 전체 기준.
            /// </summary>
            public bool EstimatedTwoHanded;

            /// <summary>검출된 구간의 시작 인덱스 (원본 배열 기준, 패딩 포함).</summary>
            public int SegmentStartIndex;

            /// <summary>검출된 구간의 종료 인덱스 (inclusive, 패딩 포함).</summary>
            public int SegmentEndIndex;

            /// <summary>디버그 메모.</summary>
            public string DebugNote;
        }

        // ─── 공개 API ────────────────────────────────────────────────────────

        /// <summary>
        /// CanonicalFrame[] 에서 동작 구간을 검출한다.
        /// </summary>
        /// <param name="frames">표준화된 프레임 배열.</param>
        /// <param name="config">검출 설정. null이면 기본값 사용.</param>
        public static SegmentResult Detect(
            CanonicalFrame[] frames,
            SegmentationConfig config = null)
        {
            if (config == null) config = SegmentationConfig.Default;

            var result = new SegmentResult();

            if (frames == null || frames.Length == 0)
            {
                result.Status = FrameQualityStatus.RecordingTooShort;
                result.DebugNote = "Empty or null input.";
                return result;
            }

            if (frames.Length < config.MinTotalFrames)
            {
                result.Status = FrameQualityStatus.RecordingTooShort;
                result.DebugNote = $"Total frames {frames.Length} < min {config.MinTotalFrames}.";
                return result;
            }

            // 손/pose 비율 계산 (전체 프레임 기준)
            int handCount = 0, poseCount = 0;
            for (int i = 0; i < frames.Length; i++)
            {
                if (frames[i].HasLeftHand || frames[i].HasRightHand) handCount++;
                if (frames[i].HasPose) poseCount++;
            }

            result.HandFrameRatio = (float)handCount / frames.Length;
            result.PoseFrameRatio = (float)poseCount / frames.Length;

            // 손 비율 검사
            if (result.HandFrameRatio < config.MinHandFrameRatio)
            {
                result.Status = FrameQualityStatus.InsufficientHandFrames;
                result.DebugNote = $"HandFrameRatio={result.HandFrameRatio:F2} < {config.MinHandFrameRatio}.";
                result.EstimatedTwoHanded = EstimateTwoHanded(frames, 0, frames.Length - 1, config);
                return result;
            }

            // Pose 비율 검사
            if (result.PoseFrameRatio < config.MinPoseFrameRatio)
            {
                result.Status = FrameQualityStatus.InsufficientPoseFrames;
                result.DebugNote = $"PoseFrameRatio={result.PoseFrameRatio:F2} < {config.MinPoseFrameRatio}.";
                result.EstimatedTwoHanded = EstimateTwoHanded(frames, 0, frames.Length - 1, config);
                return result;
            }

            // 활성 프레임 계산 (속도·에너지 기반; 손 존재 여부는 포함하지 않음)
            bool[] active = ComputeActiveFrames(frames, config);

            // 갭 연결 후 활성 구간 목록 탐색
            List<(int start, int end)> runs = FindActiveRuns(active);
            runs = MergeGaps(runs, config.MaxInactiveGapFrames);

            // 가장 긴 구간 선택
            int bestStart = -1, bestEnd = -1, bestLen = 0;
            foreach (var (s, e) in runs)
            {
                int len = e - s + 1;
                if (len > bestLen) { bestLen = len; bestStart = s; bestEnd = e; }
            }

            if (bestStart < 0 || bestLen < config.MinActiveFrames)
            {
                result.Status = FrameQualityStatus.SegmentNotFound;
                result.DebugNote = $"No active segment found (minActive={config.MinActiveFrames}, " +
                                   $"best={bestLen}, runs={runs.Count}).";
                result.EstimatedTwoHanded = EstimateTwoHanded(frames, 0, frames.Length - 1, config);
                return result;
            }

            // 패딩 적용
            int padStart = Math.Max(0, bestStart - config.StartPaddingFrames);
            int padEnd   = Math.Min(frames.Length - 1, bestEnd + config.EndPaddingFrames);

            // 최소 지속 시간 검사
            float durationMs = frames[padEnd].TimestampMs - frames[padStart].TimestampMs;
            if (durationMs < config.MinSegmentDurationMs)
            {
                result.Status = FrameQualityStatus.SegmentTooShort;
                result.DebugNote = $"Segment duration {durationMs:F0}ms < min {config.MinSegmentDurationMs}ms.";
                result.EstimatedTwoHanded = EstimateTwoHanded(frames, padStart, padEnd, config);
                return result;
            }

            // 구간 추출
            int segLen = padEnd - padStart + 1;
            var seg = new CanonicalFrame[segLen];
            Array.Copy(frames, padStart, seg, 0, segLen);

            // 양손 추정: 검출된 구간 기준
            result.EstimatedTwoHanded = EstimateTwoHanded(frames, padStart, padEnd, config);

            result.Status            = FrameQualityStatus.Analyzable;
            result.Segment           = seg;
            result.SegmentStartIndex = padStart;
            result.SegmentEndIndex   = padEnd;
            result.DebugNote         = $"Segment [{padStart}..{padEnd}], {durationMs:F0}ms, " +
                                       $"hand={result.HandFrameRatio:F2}, pose={result.PoseFrameRatio:F2}, " +
                                       $"twoHanded={result.EstimatedTwoHanded}.";
            return result;
        }

        // ─── Internal ────────────────────────────────────────────────────────

        /// <summary>
        /// 손목 이동 속도와 랜드마크 에너지를 기반으로 각 프레임의 활성 여부를 계산한다.
        /// 손 검출 여부는 속도 계산의 validity mask로만 사용하며
        /// active 판정 조건에 포함하지 않는다.
        /// </summary>
        static bool[] ComputeActiveFrames(CanonicalFrame[] frames, SegmentationConfig config)
        {
            int n = frames.Length;
            var active = new bool[n];
            var velocities = new float[n];

            // 속도 계산 (중앙 차분)
            // 손이 검출된 프레임 사이에서만 유효한 속도를 계산함
            for (int i = 1; i < n - 1; i++)
            {
                float dtMs = frames[i + 1].TimestampMs - frames[i - 1].TimestampMs;
                if (dtMs <= 0f) continue;
                float dtSec = dtMs / 1000f;

                Vector2 lPrev = frames[i - 1].LeftWristBodyRelative;
                Vector2 lNext = frames[i + 1].LeftWristBodyRelative;
                Vector2 rPrev = frames[i - 1].RightWristBodyRelative;
                Vector2 rNext = frames[i + 1].RightWristBodyRelative;

                float vl = IsValidVec2(lPrev) && IsValidVec2(lNext)
                           ? (lNext - lPrev).magnitude / dtSec
                           : 0f;
                float vr = IsValidVec2(rPrev) && IsValidVec2(rNext)
                           ? (rNext - rPrev).magnitude / dtSec
                           : 0f;

                velocities[i] = Mathf.Max(vl, vr);
            }
            if (n >= 2) { velocities[0] = velocities[1]; velocities[n - 1] = velocities[n - 2]; }

            // 랜드마크 에너지 계산
            var energies = new float[n];
            for (int i = 1; i < n; i++)
                energies[i] = ComputeLandmarkEnergy(frames[i - 1], frames[i]);

            // 스무딩 후 active 판정 (속도·에너지만 사용, 손 존재 여부 불포함)
            int w = Mathf.Max(1, config.SmoothingWindowFrames);
            for (int i = 0; i < n; i++)
            {
                int lo = Mathf.Max(0, i - w / 2);
                int hi = Mathf.Min(n - 1, i + w / 2);
                float sumV = 0f, sumE = 0f;
                for (int j = lo; j <= hi; j++) { sumV += velocities[j]; sumE += energies[j]; }
                float avg = hi - lo + 1;
                active[i] = (sumV / avg >= config.WristVelocityThreshold)
                         || (sumE / avg >= config.LandmarkEnergyThreshold);
            }

            return active;
        }

        /// <summary>active[] 에서 연속 활성 구간 목록을 추출한다.</summary>
        static List<(int start, int end)> FindActiveRuns(bool[] active)
        {
            var runs = new List<(int, int)>();
            int n = active.Length;
            int runStart = -1;
            for (int i = 0; i <= n; i++)
            {
                bool isActive = i < n && active[i];
                if (isActive && runStart < 0)
                    runStart = i;
                else if (!isActive && runStart >= 0)
                {
                    runs.Add((runStart, i - 1));
                    runStart = -1;
                }
            }
            return runs;
        }

        /// <summary>
        /// 인접 활성 구간 사이의 갭이 maxGap 이하이면 합친다.
        /// </summary>
        static List<(int start, int end)> MergeGaps(List<(int start, int end)> runs, int maxGap)
        {
            if (maxGap <= 0 || runs.Count <= 1) return runs;

            var merged = new List<(int, int)> { runs[0] };
            for (int r = 1; r < runs.Count; r++)
            {
                var (ps, pe) = merged[merged.Count - 1];
                var (cs, ce) = runs[r];
                int gap = cs - pe - 1;
                if (gap <= maxGap)
                    merged[merged.Count - 1] = (ps, ce);
                else
                    merged.Add((cs, ce));
            }
            return merged;
        }

        /// <summary>지정 구간에서 양손 수어 여부를 추정한다.</summary>
        static bool EstimateTwoHanded(CanonicalFrame[] frames, int start, int end, SegmentationConfig config)
        {
            int count = 0;
            int total = end - start + 1;
            for (int i = start; i <= end && i < frames.Length; i++)
                if (frames[i].HasLeftHand && frames[i].HasRightHand) count++;
            return total > 0 && (float)count / total > config.TwoHandedThreshold;
        }

        static float ComputeLandmarkEnergy(CanonicalFrame prev, CanonicalFrame curr)
        {
            float energy = 0f;
            energy += PointArrayEnergy(prev.RightHandPoints, curr.RightHandPoints);
            energy += PointArrayEnergy(prev.LeftHandPoints,  curr.LeftHandPoints);
            return energy;
        }

        static float PointArrayEnergy(Vector3[] a, Vector3[] b)
        {
            if (a == null || b == null || a.Length == 0 || b.Length != a.Length) return 0f;
            float sum = 0f;
            int n = Math.Min(a.Length, b.Length);
            for (int i = 0; i < n; i++)
                sum += (b[i] - a[i]).sqrMagnitude;
            return sum / n;
        }

        static bool IsValidVec2(Vector2 v)
            => !float.IsNaN(v.x) && !float.IsNaN(v.y)
            && !float.IsInfinity(v.x) && !float.IsInfinity(v.y);
    }
}
