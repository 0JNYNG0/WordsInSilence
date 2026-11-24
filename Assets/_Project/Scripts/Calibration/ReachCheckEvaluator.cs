using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// C3 동작 범위 검사: 왼팔, 오른팔, 양팔 위로.
    /// 각 포즈를 0.7~1.0초 유지하면 통과.
    /// </summary>
    public sealed class ReachCheckEvaluator
    {
        readonly RecognitionSettings _settings;
        ReachCheckPhase _phase = ReachCheckPhase.LeftArm;
        float _holdTime;
        const float RequiredHoldSeconds = 0.7f;

        public ReachCheckPhase CurrentPhase => _phase;
        public float HoldTime => _holdTime;
        public bool IsComplete => _phase == ReachCheckPhase.Done;

        // 결과 저장
        SafeFrameData _result = new SafeFrameData();
        public SafeFrameData Result => _result;

        public ReachCheckEvaluator(RecognitionSettings settings = null)
        {
            _settings = settings ?? RecognitionSettings.Default;
        }

        public void Reset()
        {
            _phase = ReachCheckPhase.LeftArm;
            _holdTime = 0f;
            _result = new SafeFrameData();
        }

        /// <summary>
        /// 매 프레임 호출. 현재 단계에 맞는 포즈가 감지되면 hold를 누적한다.
        /// </summary>
        public ReachCheckFrameResult Update(HandPoseFrame frame, float deltaTime)
        {
            var frameResult = new ReachCheckFrameResult { Phase = _phase };

            if (_phase == ReachCheckPhase.Done)
            {
                frameResult.Passed = true;
                return frameResult;
            }

            if (frame == null || !frame.HasPose || !frame.HasValidShoulders)
            {
                _holdTime = 0f;
                frameResult.Guidance = "상체가 보이도록 카메라 앞에 앉아 주세요.";
                return frameResult;
            }

            var points = frame.PoseFrame?.AllPoints;
            if (points == null || points.Length <= 16)
            {
                _holdTime = 0f;
                frameResult.Guidance = "포즈 랜드마크가 부족합니다.";
                return frameResult;
            }

            float margin = _settings.SafeFrameMarginRatio;
            bool poseOk;
            string guidance;

            switch (_phase)
            {
                case ReachCheckPhase.LeftArm:
                    poseOk = CheckLeftArmExtended(points, margin, out guidance);
                    break;
                case ReachCheckPhase.RightArm:
                    poseOk = CheckRightArmExtended(points, margin, out guidance);
                    break;
                case ReachCheckPhase.BothArmsUp:
                    poseOk = CheckBothArmsUp(points, margin, out guidance);
                    break;
                default:
                    poseOk = false;
                    guidance = null;
                    break;
            }

            if (poseOk)
            {
                _holdTime += deltaTime;
                if (_holdTime >= RequiredHoldSeconds)
                {
                    RecordPhaseResult(points, margin);
                    AdvancePhase();
                    frameResult.PhaseCompleted = true;
                    frameResult.Phase = _phase;
                    _holdTime = 0f;
                }
                frameResult.Passed = _phase == ReachCheckPhase.Done;
            }
            else
            {
                _holdTime = 0f;
                frameResult.Guidance = guidance;
            }

            frameResult.HoldTime = _holdTime;
            return frameResult;
        }

        void AdvancePhase()
        {
            switch (_phase)
            {
                case ReachCheckPhase.LeftArm:
                    _phase = ReachCheckPhase.RightArm;
                    break;
                case ReachCheckPhase.RightArm:
                    _phase = ReachCheckPhase.BothArmsUp;
                    break;
                case ReachCheckPhase.BothArmsUp:
                    _phase = ReachCheckPhase.Done;
                    _result.ReachCheckPassed = true;
                    break;
            }
        }

        void RecordPhaseResult(Vector3[] points, float margin)
        {
            switch (_phase)
            {
                case ReachCheckPhase.LeftArm:
                    // 왼쪽 손목의 x 좌표 → 왼쪽 여백
                    _result.LeftMargin = Mathf.Max(points[15].x, margin);
                    break;
                case ReachCheckPhase.RightArm:
                    // 오른쪽 손목의 x 좌표 → 오른쪽 여백 (1-x)
                    _result.RightMargin = Mathf.Max(1f - points[16].x, margin);
                    break;
                case ReachCheckPhase.BothArmsUp:
                    // 양 손목 중 더 높은(y가 작은) 값 → 상단 여백
                    float topY = Mathf.Min(points[15].y, points[16].y);
                    _result.TopMargin = Mathf.Max(topY, margin);
                    break;
            }
        }

        // MediaPipe Pose: 11=LeftShoulder, 12=RightShoulder,
        //                 13=LeftElbow, 14=RightElbow,
        //                 15=LeftWrist, 16=RightWrist

        bool CheckLeftArmExtended(Vector3[] points, float margin, out string guidance)
        {
            // 왼손목이 왼어깨보다 왼쪽에 있고, 안전 여백 내에 있는지
            Vector3 leftShoulder = points[11];
            Vector3 leftWrist = points[15];
            Vector3 leftElbow = points[13];

            if (!IsInFrame(leftWrist, margin))
            {
                guidance = "왼쪽 손이 화면 밖으로 나갔습니다. 카메라에서 조금 더 멀리 앉아 주세요.";
                return false;
            }

            if (!IsInFrame(leftElbow, margin))
            {
                guidance = "왼쪽 팔꿈치가 화면 밖입니다. 카메라에서 조금 더 멀리 앉아 주세요.";
                return false;
            }

            // 손목이 어깨보다 왼쪽(x가 작음)에 있어야 한다
            // MediaPipe: x 0=left, 1=right (화면 기준)
            if (leftWrist.x >= leftShoulder.x)
            {
                guidance = "왼팔을 편안히 왼쪽으로 뻗어 주세요.";
                return false;
            }

            guidance = null;
            return true;
        }

        bool CheckRightArmExtended(Vector3[] points, float margin, out string guidance)
        {
            Vector3 rightShoulder = points[12];
            Vector3 rightWrist = points[16];
            Vector3 rightElbow = points[14];

            if (!IsInFrame(rightWrist, margin))
            {
                guidance = "오른쪽 손이 화면 밖으로 나갔습니다. 카메라에서 조금 더 멀리 앉아 주세요.";
                return false;
            }

            if (!IsInFrame(rightElbow, margin))
            {
                guidance = "오른쪽 팔꿈치가 화면 밖입니다. 카메라에서 조금 더 멀리 앉아 주세요.";
                return false;
            }

            // 손목이 어깨보다 오른쪽(x가 큼)에 있어야 한다
            if (rightWrist.x <= rightShoulder.x)
            {
                guidance = "오른팔을 편안히 오른쪽으로 뻗어 주세요.";
                return false;
            }

            guidance = null;
            return true;
        }

        bool CheckBothArmsUp(Vector3[] points, float margin, out string guidance)
        {
            Vector3 leftWrist = points[15];
            Vector3 rightWrist = points[16];
            Vector3 nose = points[0];

            if (!IsInFrame(leftWrist, margin))
            {
                guidance = "왼손이 화면 밖입니다. 카메라를 약간 위로 올려 주세요.";
                return false;
            }

            if (!IsInFrame(rightWrist, margin))
            {
                guidance = "오른손이 화면 밖입니다. 카메라를 약간 위로 올려 주세요.";
                return false;
            }

            // 양 손목이 코보다 위(y가 작음)에 있어야 한다
            if (leftWrist.y >= nose.y || rightWrist.y >= nose.y)
            {
                guidance = "양손을 머리 위쪽으로 올려 주세요.";
                return false;
            }

            guidance = null;
            return true;
        }

        static bool IsInFrame(Vector3 point, float margin)
        {
            return point.x >= margin && point.x <= (1f - margin) &&
                   point.y >= margin && point.y <= (1f - margin);
        }
    }

    public sealed class ReachCheckFrameResult
    {
        public ReachCheckPhase Phase;
        public bool Passed;
        public bool PhaseCompleted;
        public float HoldTime;
        public string Guidance;
    }
}
