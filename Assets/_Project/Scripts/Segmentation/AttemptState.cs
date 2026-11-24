namespace WordsInSilence.Segmentation
{
    /// <summary>
    /// 수어 시도의 상태 머신 상태.
    /// </summary>
    public enum AttemptState
    {
        Inactive,
        CameraCheck,
        NeutralCalibrating,
        WaitForNeutral,
        Armed,
        Capturing,
        EndCandidate,
        Evaluating,
        Feedback,
        InvalidAttempt,
        Cancelled
    }

    /// <summary>
    /// 수어 시도의 최종 결과 상태.
    /// </summary>
    public enum AttemptResultStatus
    {
        /// <summary>임계값, 필수 특징, 후보 margin 충족.</summary>
        Passed,
        /// <summary>카메라 품질 충분, 수행 특징이 기준과 다름.</summary>
        Incorrect,
        /// <summary>가림, 화면 이탈, 긴 추적 누락, 후보 점수 근접.</summary>
        Uncertain,
        /// <summary>수행 시작 없음, 지나치게 짧거나 김, 취소.</summary>
        Invalid
    }

    /// <summary>
    /// 수어별 촬영 시점 정책.
    /// </summary>
    public enum CaptureViewPolicy
    {
        Front,
        SlightLeftPreferred,
        SlightRightPreferred,
        EitherOblique,
        MultiViewResearch
    }
}
