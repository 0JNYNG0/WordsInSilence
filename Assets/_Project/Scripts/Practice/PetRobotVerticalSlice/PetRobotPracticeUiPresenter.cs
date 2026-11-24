using System.Globalization;
using UnityEngine;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public sealed class PetRobotPracticeUiPresenter : MonoBehaviour
    {
        public const string NeedsPracticeContinueButtonLabel = "이번 단어는 연습 필요로 남기고 계속";

        [SerializeField] Rect panelRect = new Rect(24f, 24f, 620f, 720f);
        PetRobotSignPracticeSession session;
        PetRobotPracticePresenter robot;
        PetRobotCandidateCameraStartupRoute cameraRoute;
        string initializationFailure = string.Empty;
        Vector2 scroll;

        public void Bind(PetRobotSignPracticeSession practiceSession, PetRobotPracticePresenter presenter)
        {
            Bind(practiceSession, presenter, null);
        }

        public void Bind(
            PetRobotSignPracticeSession practiceSession,
            PetRobotPracticePresenter presenter,
            PetRobotCandidateCameraStartupRoute candidateCameraRoute)
        {
            session = practiceSession;
            robot = presenter;
            cameraRoute = candidateCameraRoute;
            robot?.ShowWord(session.CurrentWord);
        }

        public void ShowInitializationFailure(string message) => initializationFailure = message ?? "unknown error";

        void OnGUI()
        {
            GUILayout.BeginArea(panelRect, GUI.skin.box);
            scroll = GUILayout.BeginScrollView(scroll);
            GUILayout.Label("애완 로봇 가상 수어 연습 공간");
            if (!string.IsNullOrEmpty(initializationFailure))
            {
                GUILayout.Label("초기화 실패: " + initializationFailure);
                GUILayout.EndScrollView();
                GUILayout.EndArea();
                return;
            }
            if (session == null)
            {
                GUILayout.Label("연습 세션 준비 중...");
                GUILayout.EndScrollView();
                GUILayout.EndArea();
                return;
            }

            robot?.DrawPlaceholder(new Rect(panelRect.width - 210f, 40f, 170f, 205f));
            GUILayout.Label("로봇: " + (robot == null ? "-" : robot.Speech));
            GUILayout.Space(8f);
            GUILayout.Label("현재 목표 단어: " + session.CurrentWord);
            GUILayout.Label("진행: " + ProgressText());
            GUILayout.Label("Capture/controller 상태: " + session.CaptureState);
            GUILayout.Label("학습 상태: " + session.State);
            GUILayout.Label("Candidate route active: " + session.CandidateRouteActive);
            GUILayout.Label("Camera: " + session.CameraState);
            GUILayout.Label("Camera started / frame ready: " +
                session.CameraStarted + " / " + session.CameraFrameReady +
                " (frames " + session.CameraFrameCount + ")");
            GUILayout.Label("Provider/candidate observation ready: " +
                session.ProviderObservationReady + " (observations " +
                session.ProviderObservationCount + ")");
            if (!string.IsNullOrEmpty(session.CameraFailureCode) ||
                !string.IsNullOrEmpty(session.CameraFailureMessage))
                GUILayout.Label("Camera failure: " + session.CameraFailureCode + " / " +
                    session.CameraFailureMessage);
            if (cameraRoute != null && cameraRoute.PreviewTexture != null)
            {
                Rect previewRect = GUILayoutUtility.GetRect(360f, 240f);
                GUI.DrawTexture(previewRect, cameraRoute.PreviewTexture, ScaleMode.ScaleToFit, false);
            }

            GuidedQuestOutcome outcome = session.LastLearningOutcome;
            GUILayout.Label("학습 결과: " + (outcome == null ? "-" : outcome.Kind.ToString()));
            if (outcome != null)
            {
                GUILayout.Label("Strict accepted / expected-class / quest: " +
                    outcome.StrictAccepted + " / " + outcome.StrictExpectedWordClassAccept + " / " + outcome.StrictQuestSuccess);
            }

            // F27: shown only when the 46-point full-hand decision was the authority.
            WordsInSilence.Recognition.FullHandRuntime.FullHandGameplayDecision fullHand =
                session.LastFullHandGameplayDecision;
            if (fullHand != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("[전체 손 인식(46점) 결과 · 판정 권한 있음]");
                GUILayout.Label(WordsInSilence.Recognition.FullHandRuntime
                    .FullHandGameplayFeedback.Korean(fullHand));
                GUILayout.Label(fullHand.ProvenanceSummary());
                GUILayout.Label(WordsInSilence.Recognition.FullHandRuntime
                    .FullHandGameplayFeedback.EnrollmentPending);
            }
            GUILayout.Label("개인화 저장 상태: " + session.LastStorageResult.Status);
            GUILayout.Label("저장 상세: " + session.LastStorageResult.Message);

            CandidateCountSnapshot counts = session.CandidateCounts;
            foreach (string word in session.Words)
            {
                GUILayout.Label(session.TrustedEnrollmentEnabled
                    ? word + " 검증된 개인 표본: " + session.GetTrustedSampleCount(word) + "/3" +
                      " (ReviewOnly " + counts.ReviewOnly[word] + ")"
                    : word + " 적격 후보: " + counts.Eligible[word] + "/3" +
                      " (ReviewOnly " + counts.ReviewOnly[word] + ")");
            }
            if (session.TrustedEnrollmentEnabled)
            {
                GUILayout.Label("연습 시도: " + session.CurrentScoreableAttemptCount + "/10");
                if (session.CurrentConsecutiveTechnicalFailureCount > 0)
                    GUILayout.Label("기술 오류: " + session.CurrentConsecutiveTechnicalFailureCount + "/3");
                GUILayout.Label("개인화 상태: " + session.CurrentEnrollmentState);
                GUILayout.Label("전체 3/3/3 Gate: " + session.OverallEnrollmentGateReady);
            }
            GUILayout.Label("이번 로컬 후보 저장 수: " + counts.TotalCandidateCount);
            GUILayout.Label("세 단어 완료: " + session.AllWordsComplete);
            if (session.TrustedEnrollmentEnabled ? session.OverallEnrollmentGateReady : counts.ProfileRefreshReady)
                GUILayout.Label("개인 프로필 갱신 준비됨 (비활성: 자동 갱신/활성화 없음)");

            GUILayout.Space(10f);
            bool previousEnabled = GUI.enabled;
            GUI.enabled = previousEnabled && session.State == PetRobotPracticeState.Ready && session.CameraReady;
            if (GUILayout.Button("표현 시작")) session.StartExpression();
            GUI.enabled = previousEnabled;
            if (session.State == PetRobotPracticeState.Capturing && GUILayout.Button("표현 완료"))
            {
                session.FinishExpression();
                robot?.ShowOutcome(session.LastLearningOutcome);
            }
            if (session.CanRetry &&
                GUILayout.Button("다시 시도"))
            {
                if (session.Retry()) robot?.ShowWord(session.CurrentWord);
            }
            if (session.CanContinueWithNeedsPractice &&
                GUILayout.Button(NeedsPracticeContinueButtonLabel))
            {
                string resolvedWord = session.CurrentWord;
                if (session.ContinueWithNeedsPractice())
                {
                    if (session.State == PetRobotPracticeState.Completed)
                        robot?.ShowCompleted(session.MasteredCount, session.NeedsPracticeCount);
                    else
                    {
                        robot?.ShowNeedsPractice(resolvedWord);
                        robot?.ShowWord(session.CurrentWord);
                    }
                }
            }
            if (session.State == PetRobotPracticeState.Feedback && session.IsCurrentWordComplete &&
                GUILayout.Button(session.CurrentWord == "THANK_YOU" ? "학습 완료" : "다음 단어"))
            {
                if (session.ContinueToNextWord())
                {
                    if (session.State == PetRobotPracticeState.Completed)
                        robot?.ShowCompleted(session.MasteredCount, session.NeedsPracticeCount);
                    else robot?.ShowWord(session.CurrentWord);
                }
            }
            if (session.State == PetRobotPracticeState.Completed)
            {
                GUILayout.Space(8f);
                foreach (string word in session.Words)
                    GUILayout.Label(word + ": " + session.GetWordLearningStatus(word));
                GUILayout.Label("Mastered: " + session.MasteredCount);
                GUILayout.Label("NeedsPractice: " + session.NeedsPracticeCount);
                GUILayout.Label("Candidate saved: " + counts.TotalCandidateCount);
            }
            if (session.State == PetRobotPracticeState.Completed && GUILayout.Button("연습 공간 나가기"))
                session.ExitPractice();

            PracticeRecognitionAttempt attempt = session.LastAttempt;
            if (attempt != null && attempt.NormalizedSequence != null)
            {
                GUILayout.Space(10f);
                GUILayout.Label("진단 거리 GREETING / LOVE / THANK_YOU: " +
                    R(attempt.GreetingDistance) + " / " + R(attempt.LoveDistance) + " / " + R(attempt.ThankYouDistance));
                GUILayout.Label("승인 expected threshold: " + R(attempt.ExpectedWordThreshold));
                GUILayout.Label("Session: " + attempt.CaptureSessionId);
            }
            DrawCommonSignalDiagnostics(attempt);
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        /// <summary>
        /// Step 10P v1.0c observability: read-only on-screen view of the frozen
        /// common-2D auxiliary signal for the last finalized attempt. It renders values
        /// the approved seams already produced and changes nothing; removing this block
        /// leaves recognition, the guided outcome and every write policy unchanged.
        /// </summary>
        void DrawCommonSignalDiagnostics(PracticeRecognitionAttempt attempt)
        {
            if (attempt == null) return;
            var signal = attempt.CommonSignal;
            var evidence = attempt.Outcome == null ? null : attempt.Outcome.Common;

            GUILayout.Space(10f);
            GUILayout.Label("[공통 ML 보조신호 · 판정 권한 없음]");
            GUILayout.Label("evidence state: " +
                (evidence == null ? "Unavailable" : evidence.State.ToString()) +
                " / signal: " + (signal == null ? "none" : "present"));
            if (signal == null) return;

            GUILayout.Label("artifact: " + signal.ArtifactVersion + " " +
                (signal.ArtifactSha256.Length > 16
                    ? signal.ArtifactSha256.Substring(0, 16)
                    : signal.ArtifactSha256));
            GUILayout.Label("requested: " + signal.RequestedTarget +
                " / eligible: " + signal.Eligible +
                " / frames: " + signal.SourceFrameCount);
            if (signal.Eligible)
            {
                GUILayout.Label("P(GREETING) / P(THANK_YOU): " +
                    R(signal.GreetingProbability) + " / " + R(signal.ThankYouProbability));
                GUILayout.Label("top: " + signal.TopClass + " " + R(signal.TopProbability) +
                    " / margin: " + R(signal.ProbabilityMargin) +
                    " / decision: " + R(signal.DecisionValue));
                GUILayout.Label("expected-target support: " + signal.ExpectedTargetSupport);
            }
            else
            {
                // Step 10P v1.0c1: the screen shows the SAME sanitized text the log
                // record carries, so a filesystem location in an upstream detail is
                // never published here either. The typed reason is preserved.
                bool redacted;
                string detail = CommonSignalDiagnosticRecord.SanitizeDetail(
                    signal.FailureDetail, out redacted);
                GUILayout.Label("failure: " + signal.FailureReason + " — " + detail);
                if (redacted) GUILayout.Label("(detail redacted for logging safety)");
            }
        }

        string ProgressText()
        {
            string value = string.Empty;
            foreach (string word in session.Words)
            {
                if (value.Length > 0) value += " → ";
                value += word + " [" + session.GetWordLearningStatus(word) + "]";
            }
            return value;
        }

        static string R(double value) => value.ToString("R", CultureInfo.InvariantCulture);
    }
}
