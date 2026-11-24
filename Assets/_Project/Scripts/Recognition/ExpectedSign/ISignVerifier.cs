using WordsInSilence.Pose;

namespace WordsInSilence.Recognition.ExpectedSign
{
    /// <summary>
    /// 수어 판정 서비스 인터페이스.
    ///
    /// 게임 로직은 이 인터페이스를 통해 판정을 요청하며,
    /// 구체적인 알고리즘 선택은 구현체에 위임한다.
    /// </summary>
    public interface ISignVerifier
    {
        /// <summary>
        /// 사용자 녹화가 정답 수어와 일치하는지 판정한다.
        /// </summary>
        /// <param name="recording">사용자가 녹화한 수어 동작.</param>
        /// <param name="expectedSignId">게임이 제시한 정답 수어 ID (MotionId 또는 DisplayName).</param>
        /// <param name="mode">사용할 인식 알고리즘.</param>
        /// <returns>판정 결과. null 을 반환하지 않는다.</returns>
        SignVerificationResult Verify(
            HandPoseRecording recording,
            string expectedSignId,
            RecognitionMode mode = RecognitionMode.LegacyDtw);
    }
}
