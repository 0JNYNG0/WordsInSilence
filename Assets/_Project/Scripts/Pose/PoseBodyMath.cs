using UnityEngine;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// 상체 포즈 관련 순수 정적 수학 유틸리티.
    /// MonoBehaviour 없이 테스트 가능하도록 Unity 런타임 의존성만 유지.
    /// </summary>
    public static class PoseBodyMath
    {
        /// <summary>어깨 너비가 유효하다고 판단하는 최소값 (정규화 이미지 공간).</summary>
        public const float MinShoulderWidth = 0.05f;

        /// <summary>양 어깨의 2D 중심 좌표를 반환한다.</summary>
        public static Vector2 ShoulderCenter(Vector3 left, Vector3 right)
        {
            return new Vector2((left.x + right.x) * 0.5f, (left.y + right.y) * 0.5f);
        }

        /// <summary>양 어깨 사이의 2D 거리를 반환한다.</summary>
        public static float ShoulderWidth2D(Vector3 left, Vector3 right)
        {
            return Vector2.Distance(new Vector2(left.x, left.y), new Vector2(right.x, right.y));
        }

        /// <summary>어깨 너비가 계산에 사용하기에 충분한지 검사한다.</summary>
        public static bool IsValidShoulderWidth(float width)
        {
            return width > MinShoulderWidth && !float.IsNaN(width) && !float.IsInfinity(width);
        }

        /// <summary>
        /// 신체 기준 상대 좌표를 반환한다.
        /// center = 어깨 중심, width = 어깨 너비.
        /// 너비가 너무 작거나 NaN 이면 (float.NaN, float.NaN)을 반환한다.
        /// </summary>
        public static Vector2 BodyRelative(Vector3 point, Vector2 center, float width)
        {
            if (!IsValidShoulderWidth(width))
                return new Vector2(float.NaN, float.NaN);

            return new Vector2(
                (point.x - center.x) / width,
                (point.y - center.y) / width);
        }

        /// <summary>
        /// shoulder → elbow → wrist 가 이루는 팔 각도(도, 0~180)를 반환한다.
        /// 입력에 NaN/Infinity 가 포함된 경우 float.NaN 을 반환한다.
        /// </summary>
        public static float ArmAngleDegrees(Vector3 shoulder, Vector3 elbow, Vector3 wrist)
        {
            if (HasNaNOrInfinity3(shoulder) || HasNaNOrInfinity3(elbow) || HasNaNOrInfinity3(wrist))
                return float.NaN;

            var upperArm = new Vector2(elbow.x - shoulder.x, elbow.y - shoulder.y);
            var lowerArm = new Vector2(wrist.x - elbow.x, wrist.y - elbow.y);

            if (upperArm == Vector2.zero || lowerArm == Vector2.zero)
                return float.NaN;

            return Vector2.Angle(upperArm, lowerArm);
        }

        /// <summary>Vector2 에 NaN 또는 Infinity 성분이 포함됐는지 검사한다.</summary>
        public static bool HasNaNOrInfinity(Vector2 v)
        {
            return float.IsNaN(v.x) || float.IsNaN(v.y) ||
                   float.IsInfinity(v.x) || float.IsInfinity(v.y);
        }

        static bool HasNaNOrInfinity3(Vector3 v)
        {
            return float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                   float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z);
        }
    }
}
