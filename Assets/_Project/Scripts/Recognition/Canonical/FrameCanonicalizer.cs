using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Recognition.Canonical
{
    /// <summary>
    /// HandPoseRecording 또는 HandPoseRecordingFrame[] 을 CanonicalFrame[] 으로 변환한다.
    ///
    /// 책임:
    ///   1. 좌표 규약 변환 (body-relative X·Y 반전 등)
    ///   2. 타임스탬프 0-based 변환
    ///   3. null / 빈 배열 → null 통일
    ///   4. 이중 변환 방지
    ///   5. 어깨 폭 0 또는 NaN 프레임 처리
    ///   6. 시간 기반 리샘플링 인터페이스
    /// </summary>
    public static class FrameCanonicalizer
    {
        // ─── 최소 어깨 폭 (PoseBodyMath.MinShoulderWidth 와 동일) ────────────
        const float MinShoulderWidth = 0.05f;

        // ─── 공개 API ────────────────────────────────────────────────────────

        /// <summary>
        /// HandPoseRecording 을 CanonicalFrame[] 으로 변환한다.
        /// </summary>
        /// <param name="recording">원본 녹화 데이터. null이면 빈 배열 반환.</param>
        /// <param name="convention">입력 데이터의 좌표 규약.</param>
        public static CanonicalFrame[] Canonicalize(
            HandPoseRecording recording,
            CoordinateConvention convention)
        {
            if (recording?.Frames == null || recording.Frames.Length == 0)
                return Array.Empty<CanonicalFrame>();

            return Canonicalize(recording.Frames, convention);
        }

        /// <summary>
        /// HandPoseRecordingFrame[] 을 CanonicalFrame[] 으로 변환한다.
        /// </summary>
        public static CanonicalFrame[] Canonicalize(
            HandPoseRecordingFrame[] frames,
            CoordinateConvention convention)
        {
            if (frames == null || frames.Length == 0)
                return Array.Empty<CanonicalFrame>();

            var transform = CoordinateTransformDescriptor.For(convention);
            long firstTs = frames[0].TimestampMs;

            var result = new CanonicalFrame[frames.Length];
            for (int i = 0; i < frames.Length; i++)
                result[i] = ConvertFrame(frames[i], transform, firstTs);

            return result;
        }

        // AssertNotAlreadyCanonical は削除。
        // 이중 변환 방지는 타입 시스템으로 충분히 보장된다:
        //   Canonicalize(HandPoseRecordingFrame[], ...) 은 CanonicalFrame[] 을 받지 않으므로
        //   컴파일 타임에 실수로 CanonicalFrame[] 을 재투입할 수 없다.
        // 런타임 추적이 필요하면 CanonicalFrame.AppliedTransform.Source 를 직접 확인할 것.

        /// <summary>
        /// 시간 기반 리샘플링: targetFps 에 맞게 균일 타임스탬프 축의 새 프레임을 생성한다.
        ///
        /// 동작 방식:
        ///   - 출력 타임스탬프 = 0, step, 2·step, …, (nFrames-1)·step  (ms)
        ///   - nFrames = floor(totalMs / step) + 1  (양 끝점 포함)
        ///   - body-relative 좌표·랜드마크 포인트: 양쪽 프레임이 유효하면 선형 보간
        ///   - 한쪽이 유효하지 않으면 유효한 쪽의 값 사용; 둘 다 유효하지 않으면 NaN/null 유지
        ///   - 출력 프레임은 모두 새로 생성되며 원본 배열을 공유하지 않는다
        ///   - 단일 프레임 입력: TimestampMs=0 인 복사본 1개를 반환
        ///   - targetFps &lt;= 0: 원본 배열을 그대로 반환 (복사 없음)
        /// </summary>
        public static CanonicalFrame[] ResampleToFps(CanonicalFrame[] canonical, float targetFps)
        {
            if (canonical == null || canonical.Length == 0) return Array.Empty<CanonicalFrame>();
            if (targetFps <= 0f) return canonical;

            if (canonical.Length == 1)
                return new[] { CloneFrameAt(canonical[0], 0L) };

            // 0-based 총 구간 길이 (ms)
            long firstTs  = canonical[0].TimestampMs;
            long totalMs  = canonical[canonical.Length - 1].TimestampMs - firstTs;
            if (totalMs <= 0)
                return new[] { CloneFrameAt(canonical[0], 0L) };

            float stepMs  = 1000f / targetFps;
            // 양 끝점 포함: n = floor(total / step) + 1
            int nFrames   = Mathf.Max(2, Mathf.FloorToInt((float)totalMs / stepMs) + 1);

            var result = new CanonicalFrame[nFrames];
            for (int i = 0; i < nFrames; i++)
            {
                long outTs   = (long)Math.Round(i * stepMs);        // 출력 타임스탬프 (0-based)
                long srcTs   = firstTs + outTs;                     // 원본 절대 타임스탬프 기준
                result[i]    = InterpolateAt(canonical, srcTs, outTs, firstTs);
            }
            return result;
        }

        // ─── Internal ────────────────────────────────────────────────────────

        static CanonicalFrame ConvertFrame(
            HandPoseRecordingFrame src,
            CoordinateTransformDescriptor tr,
            long firstTimestampMs)
        {
            var f = new CanonicalFrame
            {
                OriginalTimestampMs  = src.TimestampMs,
                TimestampMs          = src.TimestampMs - firstTimestampMs,
                HasLeftHand          = src.HasLeftHand,
                HasRightHand         = src.HasRightHand,
                HasPose              = src.HasPose,
                HasValidShoulders    = src.HasValidShoulders,
                ShoulderCenter       = src.ShoulderCenter,
                ShoulderWidth        = src.ShoulderWidth,
                AppliedTransform     = tr,
                OriginalQuality      = (int)src.Quality,
            };

            // 어깨 폭 0 / NaN 프레임: body-relative 는 NaN으로 유지
            bool validShoulder = src.HasValidShoulders
                                 && src.ShoulderWidth >= MinShoulderWidth
                                 && !float.IsNaN(src.ShoulderWidth)
                                 && !float.IsInfinity(src.ShoulderWidth);

            if (validShoulder)
            {
                f.LeftWristBodyRelative  = ApplyBodyRelativeTransform(src.LeftWristBodyRelative,  tr);
                f.RightWristBodyRelative = ApplyBodyRelativeTransform(src.RightWristBodyRelative, tr);
            }
            else
            {
                f.LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN);
                f.RightWristBodyRelative = new Vector2(float.NaN, float.NaN);
            }

            // Hand landmarks: null / 빈 배열 → null 통일
            f.LeftHandPoints  = NormalizePoints(src.LeftHandPoints,  tr, screenCoords: true);
            f.RightHandPoints = NormalizePoints(src.RightHandPoints, tr, screenCoords: true);
            f.PosePoints      = NormalizePoints(src.PosePoints,      tr, screenCoords: false);

            return f;
        }

        static Vector2 ApplyBodyRelativeTransform(Vector2 v, CoordinateTransformDescriptor tr)
        {
            float x = float.IsNaN(v.x) || float.IsInfinity(v.x) ? float.NaN : v.x;
            float y = float.IsNaN(v.y) || float.IsInfinity(v.y) ? float.NaN : v.y;

            if (tr.FlipBodyRelativeX && !float.IsNaN(x)) x = -x;
            if (tr.FlipBodyRelativeY && !float.IsNaN(y)) y = -y;

            return new Vector2(x, y);
        }

        static Vector3[] NormalizePoints(Vector3[] pts, CoordinateTransformDescriptor tr, bool screenCoords)
        {
            if (pts == null || pts.Length == 0) return null;  // 빈 배열도 null로 통일

            bool flipX = screenCoords && tr.FlipHandScreenX;
            bool flipY = screenCoords && tr.FlipHandScreenY;

            if (!flipX && !flipY) return pts;  // 변환 없으면 그대로

            var result = new Vector3[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                float x = flipX ? (1f - pts[i].x) : pts[i].x;
                float y = flipY ? (1f - pts[i].y) : pts[i].y;
                result[i] = new Vector3(x, y, pts[i].z);
            }
            return result;
        }

        /// <summary>
        /// 원본 배열에서 srcTs 위치의 보간 프레임을 만든다.
        /// srcTs: 원본 절대 타임스탬프 기준.
        /// outTs: 출력 프레임에 기록할 0-based 타임스탬프.
        /// firstTs: 원본 첫 프레임의 절대 타임스탬프 (srcTs 보정용).
        /// </summary>
        static CanonicalFrame InterpolateAt(
            CanonicalFrame[] frames, long srcTs, long outTs, long firstTs)
        {
            // srcTs 를 감싸는 두 원본 프레임 인덱스를 이진 탐색으로 찾는다
            int lo = 0, hi = frames.Length - 1;

            // left: TimestampMs+firstTs <= srcTs 를 만족하는 가장 큰 인덱스
            int left = 0;
            {
                int a = 0, b = frames.Length - 1;
                while (a <= b)
                {
                    int m = (a + b) / 2;
                    if (frames[m].TimestampMs + firstTs <= srcTs) { left = m; a = m + 1; }
                    else b = m - 1;
                }
            }
            int right = Math.Min(left + 1, frames.Length - 1);

            var L = frames[left];
            var R = frames[right];

            // 보간 계수 t ∈ [0, 1]
            float t = 0f;
            long span = (R.TimestampMs + firstTs) - (L.TimestampMs + firstTs);
            if (span > 0)
                t = Math.Min(1f, (float)(srcTs - (L.TimestampMs + firstTs)) / span);

            return LerpFrames(L, R, t, outTs);
        }

        static CanonicalFrame LerpFrames(CanonicalFrame L, CanonicalFrame R, float t, long outTs)
        {
            var f = new CanonicalFrame
            {
                TimestampMs          = outTs,
                OriginalTimestampMs  = LerpLong(L.OriginalTimestampMs, R.OriginalTimestampMs, t),
                HasLeftHand          = t < 0.5f ? L.HasLeftHand  : R.HasLeftHand,
                HasRightHand         = t < 0.5f ? L.HasRightHand : R.HasRightHand,
                HasPose              = t < 0.5f ? L.HasPose      : R.HasPose,
                HasValidShoulders    = t < 0.5f ? L.HasValidShoulders : R.HasValidShoulders,
                ShoulderCenter       = Vector2.Lerp(L.ShoulderCenter, R.ShoulderCenter, t),
                ShoulderWidth        = Mathf.Lerp(L.ShoulderWidth, R.ShoulderWidth, t),
                AppliedTransform     = L.AppliedTransform,
                OriginalQuality      = t < 0.5f ? L.OriginalQuality : R.OriginalQuality,
                LeftWristBodyRelative  = LerpVec2(L.LeftWristBodyRelative,  R.LeftWristBodyRelative,  t),
                RightWristBodyRelative = LerpVec2(L.RightWristBodyRelative, R.RightWristBodyRelative, t),
                LeftHandPoints   = LerpPoints(L.LeftHandPoints,  R.LeftHandPoints,  t),
                RightHandPoints  = LerpPoints(L.RightHandPoints, R.RightHandPoints, t),
                PosePoints       = LerpPoints(L.PosePoints,      R.PosePoints,      t),
            };
            return f;
        }

        static Vector2 LerpVec2(Vector2 a, Vector2 b, float t)
        {
            bool aValid = IsValidVec2(a);
            bool bValid = IsValidVec2(b);
            if (aValid && bValid) return Vector2.Lerp(a, b, t);
            if (aValid) return a;
            if (bValid) return b;
            return new Vector2(float.NaN, float.NaN);
        }

        static Vector3[] LerpPoints(Vector3[] a, Vector3[] b, float t)
        {
            if (a == null && b == null) return null;
            if (a == null) return ClonePoints(b);
            if (b == null) return ClonePoints(a);
            if (a.Length != b.Length) return t < 0.5f ? ClonePoints(a) : ClonePoints(b);

            var result = new Vector3[a.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = Vector3.Lerp(a[i], b[i], t);
            return result;
        }

        static Vector3[] ClonePoints(Vector3[] src)
        {
            if (src == null) return null;
            var dst = new Vector3[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }

        static CanonicalFrame CloneFrameAt(CanonicalFrame src, long newTs)
        {
            return new CanonicalFrame
            {
                TimestampMs            = newTs,
                OriginalTimestampMs    = src.OriginalTimestampMs,
                HasLeftHand            = src.HasLeftHand,
                HasRightHand           = src.HasRightHand,
                HasPose                = src.HasPose,
                HasValidShoulders      = src.HasValidShoulders,
                ShoulderCenter         = src.ShoulderCenter,
                ShoulderWidth          = src.ShoulderWidth,
                AppliedTransform       = src.AppliedTransform,
                OriginalQuality        = src.OriginalQuality,
                LeftWristBodyRelative  = src.LeftWristBodyRelative,
                RightWristBodyRelative = src.RightWristBodyRelative,
                LeftHandPoints         = ClonePoints(src.LeftHandPoints),
                RightHandPoints        = ClonePoints(src.RightHandPoints),
                PosePoints             = ClonePoints(src.PosePoints),
            };
        }

        static long LerpLong(long a, long b, float t)
            => a + (long)Math.Round((b - a) * t);

        static bool IsValidVec2(Vector2 v)
            => !float.IsNaN(v.x) && !float.IsNaN(v.y)
            && !float.IsInfinity(v.x) && !float.IsInfinity(v.y);
    }
}
