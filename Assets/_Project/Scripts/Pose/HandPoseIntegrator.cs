using System;
using UnityEngine;
using WordsInSilence.Core;
using WordsInSilence.Landmarks;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// Hand Landmarker와 Pose Landmarker의 프레임을 타임스탬프 기반으로 통합한다.
    /// 통합된 HandPoseFrame을 UnifiedFrameArrived 이벤트로 발행한다.
    /// </summary>
    public sealed class HandPoseIntegrator : MonoBehaviour
    {
        [SerializeField] MediaPipeLandmarkProvider _handProvider;
        [SerializeField] PoseLandmarkProvider      _poseProvider;

        [Tooltip("Hand와 Pose 프레임을 같은 쌍으로 묶을 최대 타임스탬프 차이 (ms).")]
        [SerializeField] float _maxTimestampDiffMs = 100f;

        LandmarkFrame      _latestHandFrame;
        UpperBodyPoseFrame _latestPoseFrame;

        public event Action<HandPoseFrame> UnifiedFrameArrived;

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void Start()
        {
            if (_handProvider != null) _handProvider.FrameArrived  += OnHandFrame;
            if (_poseProvider != null) _poseProvider.PoseFrameArrived += OnPoseFrame;
        }

        void OnDestroy()
        {
            if (_handProvider != null) _handProvider.FrameArrived     -= OnHandFrame;
            if (_poseProvider != null) _poseProvider.PoseFrameArrived -= OnPoseFrame;
        }

        // ─── Event handlers ──────────────────────────────────────────────────

        void OnHandFrame(LandmarkFrame frame)
        {
            _latestHandFrame = frame;
            TryCombine();
        }

        void OnPoseFrame(UpperBodyPoseFrame frame)
        {
            _latestPoseFrame = frame;
            TryCombine();
        }

        // ─── Internal ────────────────────────────────────────────────────────

        void TryCombine()
        {
            bool hasHand = _latestHandFrame != null;
            bool hasPose = _latestPoseFrame != null;

            if (!hasHand && !hasPose) return;

            if (hasHand && !hasPose)
            {
                Emit(BuildHandOnlyFrame(_latestHandFrame));
                _latestHandFrame = null;
                return;
            }

            if (!hasHand && hasPose)
            {
                Emit(BuildPoseOnlyFrame(_latestPoseFrame));
                _latestPoseFrame = null;
                return;
            }

            // 둘 다 있음 — 타임스탬프 차이 확인
            long handMs = (long)(_latestHandFrame.Timestamp * 1000.0);
            long poseMs = _latestPoseFrame.TimestampMs;
            float diff  = Mathf.Abs(handMs - poseMs);

            if (diff > _maxTimestampDiffMs)
            {
                // 오래된 쪽 폐기 후 단일 프레임 발행
                if (handMs < poseMs)
                {
                    Emit(BuildHandOnlyFrame(_latestHandFrame));
                    _latestHandFrame = null;
                }
                else
                {
                    Emit(BuildPoseOnlyFrame(_latestPoseFrame));
                    _latestPoseFrame = null;
                }
                return;
            }

            // 통합 프레임 생성
            Emit(BuildCombinedFrame(_latestHandFrame, _latestPoseFrame, diff));
            _latestHandFrame = null;
            _latestPoseFrame = null;
        }

        HandPoseFrame BuildHandOnlyFrame(LandmarkFrame handFrame)
        {
            var (hasLeft, hasRight) = GetHandedness(handFrame);
            return new HandPoseFrame
            {
                Timestamp    = handFrame.Timestamp,
                TimestampMs  = (long)(handFrame.Timestamp * 1000.0),
                HandFrame    = handFrame,
                HasLeftHand  = hasLeft,
                HasRightHand = hasRight,
                LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
                RightWristBodyRelative = new Vector2(float.NaN, float.NaN),
                LeftArmAngleDegrees    = float.NaN,
                RightArmAngleDegrees   = float.NaN,
                LeftHandPoseWristDist  = float.NaN,
                RightHandPoseWristDist = float.NaN,
                Quality = FrameDataQuality.HandOnly
            };
        }

        HandPoseFrame BuildPoseOnlyFrame(UpperBodyPoseFrame poseFrame)
        {
            var center = PoseBodyMath.ShoulderCenter(poseFrame.LeftShoulder, poseFrame.RightShoulder);
            var width  = PoseBodyMath.ShoulderWidth2D(poseFrame.LeftShoulder, poseFrame.RightShoulder);

            return new HandPoseFrame
            {
                Timestamp         = poseFrame.Timestamp,
                TimestampMs       = poseFrame.TimestampMs,
                PoseFrame         = poseFrame,
                HasPose           = true,
                HasValidShoulders = poseFrame.HasValidShoulders,
                ShoulderCenter    = center,
                ShoulderWidth     = width,
                LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
                RightWristBodyRelative = new Vector2(float.NaN, float.NaN),
                LeftArmAngleDegrees    = float.NaN,
                RightArmAngleDegrees   = float.NaN,
                LeftHandPoseWristDist  = float.NaN,
                RightHandPoseWristDist = float.NaN,
                Quality = FrameDataQuality.PoseOnly
            };
        }

        HandPoseFrame BuildCombinedFrame(LandmarkFrame handFrame, UpperBodyPoseFrame poseFrame, float diffMs)
        {
            var (hasLeft, hasRight) = GetHandedness(handFrame);
            var center = PoseBodyMath.ShoulderCenter(poseFrame.LeftShoulder, poseFrame.RightShoulder);
            var width  = PoseBodyMath.ShoulderWidth2D(poseFrame.LeftShoulder, poseFrame.RightShoulder);
            bool validShoulders = poseFrame.HasValidShoulders && PoseBodyMath.IsValidShoulderWidth(width);

            long handMs = (long)(handFrame.Timestamp * 1000.0);

            var frame = new HandPoseFrame
            {
                Timestamp                = poseFrame.Timestamp,
                TimestampMs              = poseFrame.TimestampMs,
                HandPoseTimestampDiffMs  = diffMs,
                HandFrame                = handFrame,
                HasLeftHand              = hasLeft,
                HasRightHand             = hasRight,
                PoseFrame                = poseFrame,
                HasPose                  = true,
                HasValidShoulders        = validShoulders,
                ShoulderCenter           = center,
                ShoulderWidth            = width,
                LeftWristBodyRelative    = new Vector2(float.NaN, float.NaN),
                RightWristBodyRelative   = new Vector2(float.NaN, float.NaN),
                LeftArmAngleDegrees      = float.NaN,
                RightArmAngleDegrees     = float.NaN,
                LeftHandPoseWristDist    = float.NaN,
                RightHandPoseWristDist   = float.NaN
            };

            frame.IsValidForBodyRelativeEvaluation = validShoulders;

            if (validShoulders)
            {
                frame.LeftWristBodyRelative  = PoseBodyMath.BodyRelative(poseFrame.LeftWrist,  center, width);
                frame.RightWristBodyRelative = PoseBodyMath.BodyRelative(poseFrame.RightWrist, center, width);
                frame.LeftArmAngleDegrees    = PoseBodyMath.ArmAngleDegrees(poseFrame.LeftShoulder,  poseFrame.LeftElbow,  poseFrame.LeftWrist);
                frame.RightArmAngleDegrees   = PoseBodyMath.ArmAngleDegrees(poseFrame.RightShoulder, poseFrame.RightElbow, poseFrame.RightWrist);

                // 손목-포즈 거리 계산 (Hand의 손목 랜드마크 인덱스 = 0)
                if (handFrame.Hands != null)
                {
                    foreach (var hand in handFrame.Hands)
                    {
                        if (hand.Points == null || hand.Points.Length < 1) continue;
                        var handWrist = new Vector2(hand.Points[0].X, hand.Points[0].Y);

                        if (hand.IsLeftHand)
                        {
                            var poseWrist = new Vector2(poseFrame.LeftWrist.x, poseFrame.LeftWrist.y);
                            frame.LeftHandPoseWristDist = Vector2.Distance(handWrist, poseWrist);
                        }
                        else
                        {
                            var poseWrist = new Vector2(poseFrame.RightWrist.x, poseFrame.RightWrist.y);
                            frame.RightHandPoseWristDist = Vector2.Distance(handWrist, poseWrist);
                        }
                    }
                }
            }

            frame.Quality = DetermineQuality(frame);
            return frame;
        }

        static FrameDataQuality DetermineQuality(HandPoseFrame f)
        {
            if (!f.HasPose) return FrameDataQuality.HandOnly;
            if (!f.HasValidShoulders) return FrameDataQuality.InsufficientPose;

            bool hasAnyHand = f.HasLeftHand || f.HasRightHand;
            if (!hasAnyHand) return FrameDataQuality.PoseOnly;

            const float AssociationThreshold = 0.3f;
            bool leftUncertain  = f.HasLeftHand  && !float.IsNaN(f.LeftHandPoseWristDist)  && f.LeftHandPoseWristDist  > AssociationThreshold;
            bool rightUncertain = f.HasRightHand && !float.IsNaN(f.RightHandPoseWristDist) && f.RightHandPoseWristDist > AssociationThreshold;

            if (leftUncertain || rightUncertain) return FrameDataQuality.AssociationUncertain;

            return FrameDataQuality.Good;
        }

        static (bool hasLeft, bool hasRight) GetHandedness(LandmarkFrame frame)
        {
            bool left = false, right = false;
            if (frame?.Hands == null) return (false, false);
            foreach (var h in frame.Hands)
            {
                if (h.IsLeftHand) left = true;
                else right = true;
            }
            return (left, right);
        }

        void Emit(HandPoseFrame frame) => UnifiedFrameArrived?.Invoke(frame);
    }
}
