using UnityEngine;
using WordsInSilence.Core;

namespace WordsInSilence.Pose
{
    /// <summary>Hand+Pose 통합 프레임의 데이터 품질을 나타낸다.</summary>
    public enum FrameDataQuality
    {
        /// <summary>Hand + Pose 모두 유효, 신체 기준 계산 가능.</summary>
        Good,
        /// <summary>Pose 없음 — 기존 Hand-only 모드와 동일하게 처리.</summary>
        HandOnly,
        /// <summary>손 미검출, Pose만 존재.</summary>
        PoseOnly,
        /// <summary>Pose 존재하나 어깨 visibility 부족.</summary>
        InsufficientPose,
        /// <summary>손목-포즈 매칭이 불확실 (손 교차 등).</summary>
        AssociationUncertain,
        /// <summary>몸체 일부가 화면 밖.</summary>
        OutOfFrame,
        /// <summary>Hand도 Pose도 없음.</summary>
        Invalid
    }

    /// <summary>
    /// Hand Landmarker 결과와 Pose Landmarker 결과를 통합한 프레임.
    /// </summary>
    public sealed class HandPoseFrame
    {
        // ─── 타이밍 ────────────────────────────────────────────────────────
        public float Timestamp;
        public long  TimestampMs;
        /// <summary>Hand와 Pose 프레임의 타임스탬프 차이 (ms).</summary>
        public float HandPoseTimestampDiffMs;

        // ─── Hand ──────────────────────────────────────────────────────────
        /// <summary>Hand Landmarker 결과 (null 가능).</summary>
        public LandmarkFrame HandFrame;
        public bool HasLeftHand;
        public bool HasRightHand;

        // ─── Pose ──────────────────────────────────────────────────────────
        /// <summary>Pose Landmarker 결과 (null 가능).</summary>
        public UpperBodyPoseFrame PoseFrame;
        public bool HasPose;
        public bool HasValidShoulders;

        // ─── 신체 메트릭 ──────────────────────────────────────────────────
        public Vector2 ShoulderCenter;
        public float   ShoulderWidth;

        // ─── 신체 기준 손목 위치 ─────────────────────────────────────────
        /// <summary>어깨 기준 정규화 좌표로 계산 가능한 상태이면 true.</summary>
        public bool IsValidForBodyRelativeEvaluation;

        /// <summary>왼손목 신체 기준 좌표. 유효하지 않으면 (float.NaN, float.NaN).</summary>
        public Vector2 LeftWristBodyRelative;
        /// <summary>오른손목 신체 기준 좌표. 유효하지 않으면 (float.NaN, float.NaN).</summary>
        public Vector2 RightWristBodyRelative;

        // ─── 팔 각도 (진단용) ─────────────────────────────────────────────
        public float LeftArmAngleDegrees;
        public float RightArmAngleDegrees;

        // ─── 손목-포즈 매칭 품질 ─────────────────────────────────────────
        /// <summary>왼손 Hand 손목과 Pose 왼손목의 정규화 거리.</summary>
        public float LeftHandPoseWristDist;
        /// <summary>오른손 Hand 손목과 Pose 오른손목의 정규화 거리.</summary>
        public float RightHandPoseWristDist;

        // ─── 품질 ─────────────────────────────────────────────────────────
        public FrameDataQuality Quality;
    }
}
