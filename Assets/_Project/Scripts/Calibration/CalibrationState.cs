namespace WordsInSilence.Calibration
{
    /// <summary>
    /// C0~C5 보정 흐름의 상태.
    /// </summary>
    public enum CalibrationState
    {
        /// <summary>보정 시작 전.</summary>
        Idle,
        /// <summary>C0: 프로필 및 카메라 선택.</summary>
        ProfileAndCameraSelect,
        /// <summary>C1: 기본 프레이밍 검사 (머리, 양어깨).</summary>
        FramingCheck,
        /// <summary>C2: 신체 측정 (어깨, 팔꿈치, 손이 보이는 상태에서 비율 측정).</summary>
        BodyMeasurement,
        /// <summary>C3: 중립 자세 등록 (편안한 자세로 2~3초 유지).</summary>
        NeutralCalibration,
        /// <summary>C4: 좌/우/상단 동작 범위 검사.</summary>
        ReachCheck,
        /// <summary>C5: 보정 확인.</summary>
        Confirmation,
        /// <summary>보정 완료.</summary>
        Complete,
        /// <summary>보정 실패.</summary>
        Failed
    }

    /// <summary>
    /// C3 reach check의 하위 단계.
    /// </summary>
    public enum ReachCheckPhase
    {
        LeftArm,
        RightArm,
        BothArmsUp,
        Done
    }
}
