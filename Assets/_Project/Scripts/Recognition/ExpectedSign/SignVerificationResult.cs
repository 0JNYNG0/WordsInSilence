using WordsInSilence.Recognition.Quality;

namespace WordsInSilence.Recognition.ExpectedSign
{
    /// <summary>
    /// 수어 판정 결과.
    /// 아직 근거가 없는 점수는 float.NaN ("not_evaluated").
    /// </summary>
    public sealed class SignVerificationResult
    {
        // ─── 입력 정보 ──────────────────────────────────────────────────────

        /// <summary>게임이 제시한 정답 수어 ID.</summary>
        public string ExpectedSignId;

        /// <summary>인식 알고리즘이 예측한 (또는 가장 가까운) 수어 ID.</summary>
        public string PredictedSignId;

        /// <summary>사용된 알고리즘 모드.</summary>
        public RecognitionMode Mode;

        // ─── 품질 ──────────────────────────────────────────────────────────

        /// <summary>입력 품질 및 동작 구간 검출 상태.</summary>
        public FrameQualityStatus QualityStatus;

        // ─── 판정 ──────────────────────────────────────────────────────────

        /// <summary>정답으로 수리됐는지 여부.</summary>
        public bool Accepted;

        // ─── 점수 ──────────────────────────────────────────────────────────
        // 아직 근거가 없는 점수는 float.NaN (not_evaluated).

        /// <summary>전체 종합 점수 (0~1 또는 NaN).</summary>
        public float OverallScore = float.NaN;

        /// <summary>손목 궤적·이동 패턴 점수 (0~1 또는 NaN).</summary>
        public float MotionScore = float.NaN;

        /// <summary>손 형태(Hand shape) 점수 (0~1 또는 NaN).</summary>
        public float HandShapeScore = float.NaN;

        /// <summary>위치(Location) 점수 (0~1 또는 NaN).</summary>
        public float LocationScore = float.NaN;

        /// <summary>방향(Direction/Palm orientation) 점수 (0~1 또는 NaN).</summary>
        public float DirectionScore = float.NaN;

        // ─── 실패 정보 ──────────────────────────────────────────────────────

        /// <summary>품질 또는 인식 실패 이유. 성공 시 null.</summary>
        public string FailureReason;

        // ─── 디버그 ─────────────────────────────────────────────────────────

        /// <summary>디버그용 추가 정보 (알고리즘별 세부 수치 등).</summary>
        public DebugMetrics Debug;

        // ─── 점수 유효성 헬퍼 ───────────────────────────────────────────────

        /// <summary>OverallScore가 평가되었는지.</summary>
        public bool IsOverallEvaluated => !float.IsNaN(OverallScore);

        // ─── 팩토리 ─────────────────────────────────────────────────────────

        public static SignVerificationResult QualityFailure(
            string expectedSignId,
            FrameQualityStatus status,
            string reason,
            RecognitionMode mode)
            => new SignVerificationResult
            {
                ExpectedSignId = expectedSignId,
                Mode           = mode,
                QualityStatus  = status,
                Accepted       = false,
                FailureReason  = reason,
                Debug          = new DebugMetrics(),
            };
    }

    /// <summary>알고리즘별 디버그 수치.</summary>
    public sealed class DebugMetrics
    {
        /// <summary>총 프레임 수.</summary>
        public int TotalFrames;

        /// <summary>구간 프레임 수.</summary>
        public int SegmentFrames;

        /// <summary>손 유효 프레임 비율.</summary>
        public float HandFrameRatio = float.NaN;

        /// <summary>pose 유효 프레임 비율.</summary>
        public float PoseFrameRatio = float.NaN;

        /// <summary>DTW 점수 (LegacyDtw 모드).</summary>
        public float DtwScore = float.NaN;

        /// <summary>SVM ConfidenceMargin (LegacySvm 모드).</summary>
        public float SvmConfidenceMargin = float.NaN;

        /// <summary>ExpectedSignPrototype 원시 유사도.</summary>
        public float PrototypeSimilarity = float.NaN;

        /// <summary>적용된 좌표 변환 설명.</summary>
        public string CoordinateTransform;

        /// <summary>기타 메모.</summary>
        public string Notes;
    }
}
