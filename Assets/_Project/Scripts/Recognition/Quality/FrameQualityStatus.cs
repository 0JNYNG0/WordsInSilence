namespace WordsInSilence.Recognition.Quality
{
    /// <summary>
    /// 단일 입력 녹화에 대한 품질 검사 결과.
    /// 인식 실패와 품질 실패를 구분한다.
    /// </summary>
    public enum FrameQualityStatus
    {
        /// <summary>품질 기준을 만족하며 분석 가능.</summary>
        Analyzable,

        /// <summary>유효 손 프레임 비율이 임계값 미만.</summary>
        InsufficientHandFrames,

        /// <summary>상체 기준점(어깨)을 충분히 찾지 못함.</summary>
        InsufficientPoseFrames,

        /// <summary>동작 구간이 너무 짧음 (활성 프레임 수 부족).</summary>
        SegmentTooShort,

        /// <summary>동작 구간(시작·종료)을 찾지 못함.</summary>
        SegmentNotFound,

        /// <summary>녹화 자체가 너무 짧음 (총 프레임 부족).</summary>
        RecordingTooShort,

        /// <summary>내부 처리 오류.</summary>
        InternalError,
    }
}
