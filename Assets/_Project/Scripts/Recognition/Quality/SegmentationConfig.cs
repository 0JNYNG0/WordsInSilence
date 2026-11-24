using UnityEngine;

namespace WordsInSilence.Recognition.Quality
{
    /// <summary>
    /// 동작 구간 검출과 품질 검사에 사용되는 모든 임계값을 한 곳에서 관리한다.
    /// 코드에 흩어진 매직 상수 대신 이 ScriptableObject 또는 기본값을 사용한다.
    /// </summary>
    [CreateAssetMenu(
        menuName = "WordsInSilence/SegmentationConfig",
        fileName = "DefaultSegmentationConfig")]
    public sealed class SegmentationConfig : ScriptableObject
    {
        [Header("Quality Thresholds")]

        [Tooltip("유효 손 프레임 비율 최소값 (미만이면 InsufficientHandFrames).")]
        [Range(0f, 1f)]
        public float MinHandFrameRatio = 0.30f;

        [Tooltip("유효 pose 프레임 비율 최소값 (미만이면 InsufficientPoseFrames).")]
        [Range(0f, 1f)]
        public float MinPoseFrameRatio = 0.20f;

        [Tooltip("총 프레임 최소값 (미만이면 RecordingTooShort).")]
        public int MinTotalFrames = 10;

        [Header("Segment Detection")]

        [Tooltip("동작 시작/종료 검출에 사용할 손목 이동 속도 임계값 (body-relative units/s).")]
        public float WristVelocityThreshold = 0.05f;

        [Tooltip("동작 에너지 임계값 (랜드마크 이동 합계 기준).")]
        public float LandmarkEnergyThreshold = 0.002f;

        [Tooltip("동작 구간으로 인정하는 최소 연속 활성 프레임 수.")]
        public int MinActiveFrames = 6;

        [Tooltip("동작 구간으로 인정하는 최소 지속 시간 (ms).")]
        public float MinSegmentDurationMs = 300f;

        [Tooltip("동작 시작 전 패딩 프레임 수.")]
        public int StartPaddingFrames = 3;

        [Tooltip("동작 종료 후 패딩 프레임 수.")]
        public int EndPaddingFrames = 3;

        [Tooltip("활성 판단에 사용할 슬라이딩 윈도우 크기 (프레임).")]
        public int SmoothingWindowFrames = 3;

        [Tooltip("동작 구간 내 허용 최대 비활성 연속 프레임 수.\n" +
                 "이 이하의 짧은 정지 구간은 동작 중으로 연결한다.")]
        public int MaxInactiveGapFrames = 3;

        [Header("Hand Type")]

        [Tooltip("양손 수어 최소 양손 프레임 비율 (이 이상이면 TwoHanded).")]
        [Range(0f, 1f)]
        public float TwoHandedThreshold = 0.40f;

        // ─── 기본값 팩토리 ────────────────────────────────────────────────────

        static SegmentationConfig _default;

        /// <summary>에셋 없이 사용 가능한 기본 설정.</summary>
        public static SegmentationConfig Default
        {
            get
            {
                if (_default == null)
                {
                    _default = CreateInstance<SegmentationConfig>();
                    _default.name = "DefaultSegmentationConfig (runtime)";
                }
                return _default;
            }
        }
    }
}
