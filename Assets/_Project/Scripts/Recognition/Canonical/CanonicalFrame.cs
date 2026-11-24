using UnityEngine;

namespace WordsInSilence.Recognition.Canonical
{
    /// <summary>
    /// 좌표 표준화 이후의 단일 프레임 데이터.
    ///
    /// 표준(Canonical) 규약:
    ///   - TimestampMs: 0-based (시퀀스 첫 프레임 = 0)
    ///   - LeftWristBodyRelative.x < 0, RightWristBodyRelative.x > 0
    ///   - WristBodyRelative.y: 아래 방향이 양수 (Unity 스크린 좌표)
    ///   - Hand landmark: MediaPipe NormalizedLandmark 스크린 좌표 (x,y 0~1)
    ///   - Hand z: 손목 기준 상대 깊이 (camera=양수 방향)
    ///   - 미검출 손: null (빈 배열과 null을 모두 null로 통일)
    ///   - AppliedTransform: 이 프레임에 적용된 변환 기술자 (이중 변환 방지용)
    /// </summary>
    public sealed class CanonicalFrame
    {
        /// <summary>시퀀스 내 상대 타임스탬프 (ms). 첫 프레임이 0.</summary>
        public long TimestampMs;

        public bool HasLeftHand;
        public bool HasRightHand;
        public bool HasPose;
        public bool HasValidShoulders;

        /// <summary>왼손 21개 랜드마크. 미검출이면 null (빈 배열 아님).</summary>
        public Vector3[] LeftHandPoints;

        /// <summary>오른손 21개 랜드마크. 미검출이면 null.</summary>
        public Vector3[] RightHandPoints;

        /// <summary>Pose 33개 랜드마크. 미검출이면 null.</summary>
        public Vector3[] PosePoints;

        /// <summary>어깨 중심 (screen 좌표, 0~1 normalized).</summary>
        public Vector2 ShoulderCenter;

        /// <summary>어깨 폭 (screen 좌표 단위).</summary>
        public float ShoulderWidth;

        /// <summary>
        /// 왼손목 body-relative 좌표 (표준 규약 적용 후).
        /// x < 0 = 몸의 왼쪽, y > 0 = 몸의 아래쪽.
        /// 유효하지 않으면 NaN.
        /// </summary>
        public Vector2 LeftWristBodyRelative;

        /// <summary>
        /// 오른손목 body-relative 좌표 (표준 규약 적용 후).
        /// x > 0 = 몸의 오른쪽.
        /// </summary>
        public Vector2 RightWristBodyRelative;

        /// <summary>이 프레임에 적용된 좌표 변환 정보.</summary>
        public CoordinateTransformDescriptor AppliedTransform;

        /// <summary>원본 프레임 품질 (그대로 보존).</summary>
        public int OriginalQuality;

        /// <summary>원본 소스의 절대 타임스탬프 (보존용).</summary>
        public long OriginalTimestampMs;
    }
}
