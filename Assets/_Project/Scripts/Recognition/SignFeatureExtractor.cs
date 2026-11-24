using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// 캡처된 프레임 시퀀스에서 요소별 특징을 추출한다.
    /// 몸 기준 좌표 정규화 → 손모양, 위치, 방향, 궤적, 양손 관계.
    /// </summary>
    public static class SignFeatureExtractor
    {
        /// <summary>
        /// 프레임 시퀀스에서 요소별 특징을 추출한다.
        /// </summary>
        public static SignFeatureSet Extract(IList<HandPoseFrame> frames)
        {
            var result = new SignFeatureSet();
            if (frames == null || frames.Count < 3)
            {
                result.Valid = false;
                return result;
            }

            result.FrameCount = frames.Count;
            result.DurationMs = frames[frames.Count - 1].TimestampMs - frames[0].TimestampMs;
            result.Valid = true;

            // 결측 마스크
            int handFrames = 0, poseFrames = 0;
            int leftHandFrames = 0, rightHandFrames = 0;
            foreach (var f in frames)
            {
                if (f.HasLeftHand || f.HasRightHand) handFrames++;
                if (f.HasLeftHand) leftHandFrames++;
                if (f.HasRightHand) rightHandFrames++;
                if (f.HasPose) poseFrames++;
            }
            result.HandFrameRatio = (float)handFrames / frames.Count;
            result.PoseFrameRatio = (float)poseFrames / frames.Count;
            result.LeftHandFrameRatio = (float)leftHandFrames / frames.Count;
            result.RightHandFrameRatio = (float)rightHandFrames / frames.Count;

            // 손모양 특징 (중간 프레임 샘플링)
            result.Handshape = ExtractHandshapeFeatures(frames);

            // 몸 기준 위치 특징
            result.Position = ExtractPositionFeatures(frames);

            // 궤적 특징
            result.Trajectory = ExtractTrajectoryFeatures(frames);

            // 양손 관계
            result.TwoHandRelation = ExtractTwoHandFeatures(frames);

            return result;
        }

        /// <summary>
        /// HandPoseRecordingFrame 배열에서도 추출 가능하도록 변환.
        /// </summary>
        public static SignFeatureSet ExtractFromRecording(HandPoseRecordingFrame[] frames)
        {
            if (frames == null || frames.Length < 3) return new SignFeatureSet { Valid = false };

            var converted = new List<HandPoseFrame>(frames.Length);
            foreach (var f in frames)
            {
                converted.Add(new HandPoseFrame
                {
                    TimestampMs = f.TimestampMs,
                    HasLeftHand = f.HasLeftHand,
                    HasRightHand = f.HasRightHand,
                    HasPose = f.HasPose,
                    HasValidShoulders = f.HasValidShoulders,
                    ShoulderCenter = f.ShoulderCenter,
                    ShoulderWidth = f.ShoulderWidth,
                    LeftWristBodyRelative = f.LeftWristBodyRelative,
                    RightWristBodyRelative = f.RightWristBodyRelative,
                    IsValidForBodyRelativeEvaluation = f.HasPose && f.HasValidShoulders,
                    PoseFrame = f.PosePoints != null ? new UpperBodyPoseFrame
                    {
                        AllPoints = f.PosePoints,
                        Visibility = new float[33],
                        TimestampMs = f.TimestampMs
                    } : null,
                });
            }
            return Extract(converted);
        }

        // ─── 손모양 ────────────────────────────────────────────────────────

        static HandshapeFeatures ExtractHandshapeFeatures(IList<HandPoseFrame> frames)
        {
            var result = new HandshapeFeatures();
            int sampleCount = 0;
            float totalFingerCurl = 0f;
            float totalThumbDist = 0f;

            // 시퀀스 전체에서 중앙 50% 구간 샘플링
            int start = frames.Count / 4;
            int end = frames.Count * 3 / 4;

            for (int i = start; i < end; i++)
            {
                var f = frames[i];
                // 오른손 우선, 없으면 왼손
                Vector3[] handPts = null;
                if (f.HasRightHand && f.PoseFrame?.AllPoints != null)
                {
                    // HandFrame에서 hand points 접근 — PoseFrame의 hand point가 아닌
                    // HandPoseFrame은 HandFrame을 가지고 있지만 직접 접근이 제한적
                    // body-relative wrist로 대체 추정
                }

                if (f.IsValidForBodyRelativeEvaluation)
                {
                    sampleCount++;
                }
            }

            result.SampleCount = sampleCount;
            return result;
        }

        // ─── 위치 ──────────────────────────────────────────────────────────

        static PositionFeatures ExtractPositionFeatures(IList<HandPoseFrame> frames)
        {
            var result = new PositionFeatures();

            var leftPositions = new List<Vector2>();
            var rightPositions = new List<Vector2>();

            foreach (var f in frames)
            {
                if (!f.IsValidForBodyRelativeEvaluation) continue;

                if (f.HasLeftHand && !float.IsNaN(f.LeftWristBodyRelative.x))
                    leftPositions.Add(f.LeftWristBodyRelative);
                if (f.HasRightHand && !float.IsNaN(f.RightWristBodyRelative.x))
                    rightPositions.Add(f.RightWristBodyRelative);
            }

            if (leftPositions.Count > 0)
            {
                result.HasLeft = true;
                result.LeftMeanPosition = Mean(leftPositions);
                result.LeftPositionStd = Std(leftPositions, result.LeftMeanPosition);
            }

            if (rightPositions.Count > 0)
            {
                result.HasRight = true;
                result.RightMeanPosition = Mean(rightPositions);
                result.RightPositionStd = Std(rightPositions, result.RightMeanPosition);
            }

            return result;
        }

        // ─── 궤적 ──────────────────────────────────────────────────────────

        static TrajectoryFeatures ExtractTrajectoryFeatures(IList<HandPoseFrame> frames)
        {
            var result = new TrajectoryFeatures();

            float totalDisplacement = 0f;
            int directionChangesX = 0, directionChangesY = 0;
            Vector2 prev = Vector2.zero;
            float prevDx = 0f, prevDy = 0f;
            bool hasPrev = false;
            Vector2 firstPos = Vector2.zero, lastPos = Vector2.zero;
            bool hasFirst = false;
            float maxY = float.MinValue, minY = float.MaxValue;

            foreach (var f in frames)
            {
                if (!f.IsValidForBodyRelativeEvaluation) continue;

                // 오른손 우선
                Vector2 pos = f.HasRightHand && !float.IsNaN(f.RightWristBodyRelative.x)
                    ? f.RightWristBodyRelative
                    : f.HasLeftHand && !float.IsNaN(f.LeftWristBodyRelative.x)
                        ? f.LeftWristBodyRelative
                        : Vector2.zero;

                if (pos == Vector2.zero) continue;

                if (!hasFirst) { firstPos = pos; hasFirst = true; }
                lastPos = pos;

                if (pos.y > maxY) maxY = pos.y;
                if (pos.y < minY) minY = pos.y;

                if (hasPrev)
                {
                    float dx = pos.x - prev.x;
                    float dy = pos.y - prev.y;
                    totalDisplacement += Vector2.Distance(pos, prev);

                    if (prevDx != 0 && dx * prevDx < 0) directionChangesX++;
                    if (prevDy != 0 && dy * prevDy < 0) directionChangesY++;

                    prevDx = dx != 0 ? dx : prevDx;
                    prevDy = dy != 0 ? dy : prevDy;
                }

                prev = pos;
                hasPrev = true;
            }

            if (hasFirst)
            {
                result.Valid = true;
                result.TotalDisplacement = totalDisplacement;
                result.NetDisplacement = Vector2.Distance(firstPos, lastPos);
                result.DirectionChangesX = directionChangesX;
                result.DirectionChangesY = directionChangesY;
                result.YRange = maxY - minY;

                // 경로 폐합도: 0=완전 복귀, 1=직선
                result.PathClosure = totalDisplacement > 0.01f
                    ? result.NetDisplacement / totalDisplacement
                    : 0f;
            }

            return result;
        }

        // ─── 양손 관계 ─────────────────────────────────────────────────────

        static TwoHandFeatures ExtractTwoHandFeatures(IList<HandPoseFrame> frames)
        {
            var result = new TwoHandFeatures();
            var distances = new List<float>();
            var heightDiffs = new List<float>();

            foreach (var f in frames)
            {
                if (!f.IsValidForBodyRelativeEvaluation) continue;
                if (!f.HasLeftHand || !f.HasRightHand) continue;
                if (float.IsNaN(f.LeftWristBodyRelative.x) || float.IsNaN(f.RightWristBodyRelative.x))
                    continue;

                float dist = Vector2.Distance(f.LeftWristBodyRelative, f.RightWristBodyRelative);
                distances.Add(dist);
                heightDiffs.Add(f.LeftWristBodyRelative.y - f.RightWristBodyRelative.y);
            }

            if (distances.Count >= 3)
            {
                result.Valid = true;
                result.BothHandsFrameCount = distances.Count;
                result.MeanDistance = MeanF(distances);
                result.MinDistance = MinF(distances);
                result.MaxDistance = MaxF(distances);
                result.MeanHeightDiff = MeanF(heightDiffs);
            }

            return result;
        }

        // ─── 통계 유틸 ─────────────────────────────────────────────────────

        static Vector2 Mean(List<Vector2> pts)
        {
            Vector2 sum = Vector2.zero;
            foreach (var p in pts) sum += p;
            return sum / pts.Count;
        }

        static float Std(List<Vector2> pts, Vector2 mean)
        {
            float sum = 0f;
            foreach (var p in pts) sum += (p - mean).sqrMagnitude;
            return Mathf.Sqrt(sum / pts.Count);
        }

        static float MeanF(List<float> vals)
        {
            float sum = 0f;
            foreach (var v in vals) sum += v;
            return sum / vals.Count;
        }

        static float MinF(List<float> vals)
        {
            float m = float.MaxValue;
            foreach (var v in vals) if (v < m) m = v;
            return m;
        }

        static float MaxF(List<float> vals)
        {
            float m = float.MinValue;
            foreach (var v in vals) if (v > m) m = v;
            return m;
        }
    }

    // ─── 특징 데이터 클래스 ────────────────────────────────────────────

    public sealed class SignFeatureSet
    {
        public bool Valid;
        public int FrameCount;
        public long DurationMs;

        // 결측 마스크
        public float HandFrameRatio;
        public float PoseFrameRatio;
        public float LeftHandFrameRatio;
        public float RightHandFrameRatio;

        // 요소별 특징
        public HandshapeFeatures Handshape = new HandshapeFeatures();
        public PositionFeatures Position = new PositionFeatures();
        public TrajectoryFeatures Trajectory = new TrajectoryFeatures();
        public TwoHandFeatures TwoHandRelation = new TwoHandFeatures();
    }

    public sealed class HandshapeFeatures
    {
        public int SampleCount;
    }

    public sealed class PositionFeatures
    {
        public bool HasLeft;
        public Vector2 LeftMeanPosition;
        public float LeftPositionStd;

        public bool HasRight;
        public Vector2 RightMeanPosition;
        public float RightPositionStd;
    }

    public sealed class TrajectoryFeatures
    {
        public bool Valid;
        public float TotalDisplacement;
        public float NetDisplacement;
        public int DirectionChangesX;
        public int DirectionChangesY;
        public float YRange;
        public float PathClosure;
    }

    public sealed class TwoHandFeatures
    {
        public bool Valid;
        public int BothHandsFrameCount;
        public float MeanDistance;
        public float MinDistance;
        public float MaxDistance;
        public float MeanHeightDiff;
    }
}
