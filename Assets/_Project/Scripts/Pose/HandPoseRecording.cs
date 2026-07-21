using System;
using UnityEngine;

namespace WordsInSilence.Pose
{
    // Phase 3.5: データモデル定義のみ。
    // 保存/ロード UI は Phase 4 で追加する。
    // schemaVersion 1 (既存 SignMotionSample) は別タイプとして互換性を維持する。

    /// <summary>
    /// Hand+Pose 통합 녹화의 단일 프레임 데이터 (schemaVersion 2).
    /// </summary>
    [Serializable]
    public sealed class HandPoseRecordingFrame
    {
        public long  TimestampMs;
        public bool  HasLeftHand;
        public bool  HasRightHand;
        public bool  HasPose;
        public bool  HasValidShoulders;

        /// <summary>왼손 21개 랜드마크 (없으면 null).</summary>
        public Vector3[] LeftHandPoints;
        /// <summary>오른손 21개 랜드마크 (없으면 null).</summary>
        public Vector3[] RightHandPoints;
        /// <summary>Pose 33개 랜드마크 (없으면 null).</summary>
        public Vector3[] PosePoints;

        public Vector2 ShoulderCenter;
        public float   ShoulderWidth;

        public Vector2 LeftWristBodyRelative;
        public Vector2 RightWristBodyRelative;

        public FrameDataQuality Quality;
    }

    /// <summary>
    /// Hand+Pose 통합 녹화 세션 데이터 (schemaVersion 2).
    /// </summary>
    [Serializable]
    public sealed class HandPoseRecording
    {
        public int    SchemaVersion = 2;
        public string MotionId;
        public string DisplayName;
        public float  DurationMs;

        /// <summary>"hand_only" 또는 "hand_pose".</summary>
        public string SourceType;

        public HandPoseRecordingFrame[] Frames;

        public int TotalFrames;
        public int ValidHandFrames;
        public int ValidPoseFrames;
        public int ValidCombinedFrames;
    }
}
