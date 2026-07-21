using UnityEngine;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// MediaPipe Pose Landmarker 결과에서 추출한 상체 포즈 프레임.
    /// AllPoints: 33개 정규화 이미지 좌표 (x,y ∈ [0,1], z = 깊이 추정).
    /// </summary>
    public sealed class UpperBodyPoseFrame
    {
        /// <summary>Time.realtimeSinceStartup 기준 타임스탬프.</summary>
        public float Timestamp;

        /// <summary>밀리초 단위 타임스탬프 (VIDEO 모드용).</summary>
        public long TimestampMs;

        /// <summary>33개 정규화 이미지 좌표. x,y ∈ [0,1], z = 상대 깊이.</summary>
        public Vector3[] AllPoints;

        /// <summary>33개 visibility 점수 (0~1).</summary>
        public float[] Visibility;

        // ─── 랜드마크 인덱스 상수 (MediaPipe Pose 정의) ──────────────────────
        const int IdxNose           = 0;
        const int IdxLeftShoulder   = 11;
        const int IdxRightShoulder  = 12;
        const int IdxLeftElbow      = 13;
        const int IdxRightElbow     = 14;
        const int IdxLeftWrist      = 15;
        const int IdxRightWrist     = 16;

        public const float MinVisibilityThreshold = 0.5f;

        // ─── 캐시 프로퍼티 ────────────────────────────────────────────────────
        public Vector3 Nose          => AllPoints[IdxNose];
        public Vector3 LeftShoulder  => AllPoints[IdxLeftShoulder];
        public Vector3 RightShoulder => AllPoints[IdxRightShoulder];
        public Vector3 LeftElbow     => AllPoints[IdxLeftElbow];
        public Vector3 RightElbow    => AllPoints[IdxRightElbow];
        public Vector3 LeftWrist     => AllPoints[IdxLeftWrist];
        public Vector3 RightWrist    => AllPoints[IdxRightWrist];

        public float LeftShoulderVisibility  => Visibility[IdxLeftShoulder];
        public float RightShoulderVisibility => Visibility[IdxRightShoulder];

        /// <summary>양 어깨 visibility 가 모두 임계값 이상이면 true.</summary>
        public bool HasValidShoulders =>
            LeftShoulderVisibility  >= MinVisibilityThreshold &&
            RightShoulderVisibility >= MinVisibilityThreshold;
    }
}
