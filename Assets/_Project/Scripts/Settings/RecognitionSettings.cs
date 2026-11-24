using UnityEngine;

namespace WordsInSilence.Settings
{
    /// <summary>
    /// 실시간 수어 인식의 모든 런타임 임계값을 관리한다.
    /// Inspector에서 조정 가능한 ScriptableObject.
    /// </summary>
    [CreateAssetMenu(
        menuName = "WordsInSilence/RecognitionSettings",
        fileName = "DefaultRecognitionSettings")]
    public sealed class RecognitionSettings : ScriptableObject
    {
        [Header("Neutral Pose Detection")]

        [Tooltip("Armed 진입에 필요한 중립 유지 시간 (초).")]
        public float NeutralHoldForArmed = 1.5f;

        [Tooltip("수행 종료 확정에 필요한 중립 유지 시간 (초).")]
        public float NeutralHoldForEnd = 3.0f;

        [Tooltip("Armed 상태에서 움직임 없이 대기하는 최대 시간 (초).")]
        public float ArmedTimeoutSeconds = 6.0f;

        [Tooltip("최대 수행 시간 (초). 초과 시 캡처 종료 후 평가.")]
        public float MaxCaptureDurationSeconds = 10.0f;

        [Tooltip("시작 전 순환 버퍼 크기 (초). 첫 움직임이 잘리지 않도록.")]
        public float PreCaptureBufferSeconds = 0.35f;

        [Tooltip("종료 후 여유 프레임 (초). 자연스러운 복귀 일부 보존.")]
        public float PostCaptureMarginSeconds = 0.25f;

        [Tooltip("캡처 시작 후 중립 무시 시간 (초). 손 올리는 과정에서 즉시 종료 방지.")]
        public float CaptureGracePeriodSeconds = 0.5f;

        [Tooltip("중립 판정: 상완(어깨→팔꿈치) 방향이 수직에서 벗어날 수 있는 최대 각도 (도). 0=수직, 90=수평.")]
        public float NeutralArmVerticalMax = 45f;

        [Header("Motion Detection")]

        [Tooltip("움직임 시작 판단 임계값 (body-relative displacement).")]
        public float MotionOnsetThreshold = 0.03f;

        [Tooltip("움직임 시작 판단에 필요한 최소 연속 프레임.")]
        public int MotionOnsetMinFrames = 3;

        [Tooltip("중립 복귀 판단 임계값 (body-relative displacement).")]
        public float NeutralReturnThreshold = 0.02f;

        [Tooltip("순간적인 추적 실패 허용 시간 (초).")]
        public float TrackingGapToleranceSeconds = 0.3f;

        [Header("Calibration — Body Measurement")]

        [Tooltip("신체 측정에 필요한 최소 프레임 수.")]
        public int BodyMeasurementMinFrames = 60;

        [Tooltip("신체 측정 유지 시간 (초).")]
        public float BodyMeasurementHoldSeconds = 2.5f;

        [Tooltip("신체 측정 최대 재시도 횟수.")]
        public int BodyMeasurementMaxRetries = 3;

        [Header("Calibration — Neutral Pose")]

        [Tooltip("중립 보정 시 유효 프레임 비율 최소값.")]
        [Range(0f, 1f)]
        public float NeutralCalibrationMinValidRatio = 0.80f;

        [Tooltip("중립 자세 등록 수집 시간 (초).")]
        public float NeutralCalibrationHoldSeconds = 2.5f;

        [Tooltip("중립 전환 대기 시간 (초). 사용자가 팔을 내릴 시간.")]
        public float NeutralTransitionWaitSeconds = 3.0f;

        [Tooltip("안전 영역 좌·우·상단 여백 비율.")]
        [Range(0f, 0.2f)]
        public float SafeFrameMarginRatio = 0.06f;

        [Header("Recognition")]

        [Tooltip("통과 최소 점수 (0~1).")]
        [Range(0f, 1f)]
        public float PassThreshold = 0.60f;

        [Tooltip("공용 점수 절대 하한. 개인 점수가 높아도 이 미만이면 거부.")]
        [Range(0f, 1f)]
        public float GlobalScoreFloor = 0.35f;

        [Tooltip("1위와 2위 점수 차이가 이 미만이면 uncertain.")]
        public float AmbiguityGap = 0.04f;

        [Header("Personalization Weights")]

        [Tooltip("개인 승인 표본 0~2개: 공용 가중치.")]
        public float GlobalWeight_0to2 = 1.00f;
        [Tooltip("개인 승인 표본 3~4개: 공용 가중치.")]
        public float GlobalWeight_3to4 = 0.75f;
        [Tooltip("개인 승인 표본 5~9개: 공용 가중치.")]
        public float GlobalWeight_5to9 = 0.65f;
        [Tooltip("개인 승인 표본 10+개: 공용 가중치.")]
        public float GlobalWeight_10plus = 0.60f;

        [Header("Enrollment")]

        [Tooltip("개인 등록에 필요한 최소 승인 반복 횟수.")]
        public int EnrollmentRequiredReps = 3;

        [Tooltip("한 등록 라운드 최대 시도 횟수.")]
        public int EnrollmentMaxAttempts = 5;

        [Tooltip("등록 시 공용 레퍼런스 점수 하한.")]
        [Range(0f, 1f)]
        public float EnrollmentGlobalScoreMin = 0.45f;

        [Tooltip("수어별 최대 신뢰 표본 수.")]
        public int MaxPersonalExemplars = 20;

        // ─── Default factory ───────────────────────────────────────────────

        static RecognitionSettings _default;

        public static RecognitionSettings Default
        {
            get
            {
                if (_default == null)
                {
                    _default = CreateInstance<RecognitionSettings>();
                    _default.name = "DefaultRecognitionSettings (runtime)";
                }
                return _default;
            }
        }

        /// <summary>
        /// 개인 승인 표본 수에 따른 공용/개인 가중치를 반환한다.
        /// </summary>
        public void GetWeights(int personalExemplarCount,
                               out float globalWeight, out float personalWeight)
        {
            if (personalExemplarCount >= 10)
                globalWeight = GlobalWeight_10plus;
            else if (personalExemplarCount >= 5)
                globalWeight = GlobalWeight_5to9;
            else if (personalExemplarCount >= 3)
                globalWeight = GlobalWeight_3to4;
            else
                globalWeight = GlobalWeight_0to2;

            personalWeight = 1f - globalWeight;
        }
    }
}
