using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// 카메라 프레이밍과 추적 품질을 평가한다.
    /// C1 단계에서 사용: 머리, 양어깨, 양팔꿈치가 안정적으로 보이는지 확인.
    /// </summary>
    public sealed class CameraReadinessEvaluator
    {
        readonly RecognitionSettings _settings;

        public CameraReadinessEvaluator(RecognitionSettings settings = null)
        {
            _settings = settings ?? RecognitionSettings.Default;
        }

        /// <summary>
        /// 단일 프레임의 프레이밍 품질을 평가한다.
        /// </summary>
        public FramingResult EvaluateFraming(HandPoseFrame frame)
        {
            var result = new FramingResult();

            if (frame == null || !frame.HasPose)
            {
                result.Status = FramingStatus.NoPoseDetected;
                result.Guidance = "카메라 앞에서 상체가 보이도록 앉아 주세요.";
                return result;
            }

            if (!frame.HasValidShoulders || frame.ShoulderWidth < 0.02f)
            {
                result.Status = FramingStatus.ShouldersNotVisible;
                result.Guidance = "양쪽 어깨가 모두 보이도록 카메라를 조절해 주세요.";
                return result;
            }

            // Check landmark positions are within safe margins
            float margin = _settings.SafeFrameMarginRatio;
            var posePoints = frame.PoseFrame?.AllPoints;
            if (posePoints == null || posePoints.Length < 14)
            {
                result.Status = FramingStatus.InsufficientLandmarks;
                result.Guidance = "상체가 잘 보이도록 카메라를 조절해 주세요.";
                return result;
            }

            // Nose (0), Left Shoulder (11), Right Shoulder (12),
            // Left Elbow (13), Right Elbow (14)
            // MediaPipe Pose normalized coords: x=0~1 (left→right), y=0~1 (top→bottom)
            bool noseVisible = IsInFrame(posePoints[0], margin);
            bool leftShoulderVisible = IsInFrame(posePoints[11], margin);
            bool rightShoulderVisible = IsInFrame(posePoints[12], margin);
            bool leftElbowVisible = IsInFrame(posePoints[13], margin);
            bool rightElbowVisible = IsInFrame(posePoints[14], margin);

            result.NoseVisible = noseVisible;
            result.LeftShoulderVisible = leftShoulderVisible;
            result.RightShoulderVisible = rightShoulderVisible;
            result.LeftElbowVisible = leftElbowVisible;
            result.RightElbowVisible = rightElbowVisible;

            if (!leftShoulderVisible || !rightShoulderVisible)
            {
                result.Status = FramingStatus.ShouldersOutOfFrame;
                result.Guidance = "양쪽 어깨가 화면 안에 보이도록 조금 뒤로 이동해 주세요.";
                return result;
            }

            if (!noseVisible)
            {
                result.Status = FramingStatus.HeadOutOfFrame;
                result.Guidance = "머리가 화면 안에 보이도록 카메라를 조절해 주세요.";
                return result;
            }

            // 팔꿈치: 보이면 좋지만 필수는 아님
            // 노트북 웹캠 상반신 촬영에서 자연스럽게 앉으면
            // 팔꿈치가 화면 하단에서 잘릴 수 있음
            // (경고만 표시, 통과 차단하지 않음)

            // Shoulder width too small → too far from camera
            if (frame.ShoulderWidth < 0.08f)
            {
                result.Status = FramingStatus.TooFar;
                result.Guidance = "카메라에 조금 더 가까이 앉아 주세요.";
                return result;
            }

            // Shoulder width too large → too close
            if (frame.ShoulderWidth > 0.50f)
            {
                result.Status = FramingStatus.TooClose;
                result.Guidance = "카메라에서 조금 더 멀리 앉아 주세요.";
                return result;
            }

            result.Status = FramingStatus.Good;
            result.Guidance = null;
            return result;
        }

        /// <summary>
        /// reach check용: 특정 관절이 안전 여백 내에 있는지 확인.
        /// </summary>
        public bool IsLandmarkInSafeZone(Vector2 normalizedPoint, float margin)
        {
            return IsInFrame(normalizedPoint, margin);
        }

        static bool IsInFrame(Vector2 point, float margin)
        {
            return point.x >= margin && point.x <= (1f - margin) &&
                   point.y >= margin && point.y <= (1f - margin);
        }

        static bool IsInFrame(Vector3 point, float margin)
        {
            return point.x >= margin && point.x <= (1f - margin) &&
                   point.y >= margin && point.y <= (1f - margin);
        }
    }

    public enum FramingStatus
    {
        Good,
        NoPoseDetected,
        ShouldersNotVisible,
        ShouldersOutOfFrame,
        HeadOutOfFrame,
        ElbowsOutOfFrame,
        TooFar,
        TooClose,
        InsufficientLandmarks
    }

    public sealed class FramingResult
    {
        public FramingStatus Status;
        public string Guidance;
        public bool NoseVisible;
        public bool LeftShoulderVisible;
        public bool RightShoulderVisible;
        public bool LeftElbowVisible;
        public bool RightElbowVisible;

        public bool IsGood => Status == FramingStatus.Good;
    }
}
