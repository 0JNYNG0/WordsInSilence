using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Recognition;
using WordsInSilence.Settings;

namespace WordsInSilence.Personalization
{
    /// <summary>
    /// 개인 수어 등록·판정 서비스.
    /// - 3회 승인 등록 (공용 하한 + 품질 게이트)
    /// - 개인 템플릿 저장·로드
    /// - 공용/개인 가중치 혼합 점수
    /// </summary>
    public sealed class PersonalizationService
    {
        readonly RecognitionSettings _settings;
        readonly string _profileId;

        // 메모리 캐시: signId → enrollment
        readonly Dictionary<string, PersonalSignEnrollment> _enrollments
            = new Dictionary<string, PersonalSignEnrollment>();

        // 개인 템플릿: signId → List<PersonalExemplar>
        readonly Dictionary<string, List<PersonalExemplar>> _exemplars
            = new Dictionary<string, List<PersonalExemplar>>();

        public PersonalizationService(string profileId, RecognitionSettings settings = null)
        {
            _profileId = profileId;
            _settings = settings ?? RecognitionSettings.Default;
        }

        // ─── 등록 상태 ─────────────────────────────────────────────────────

        public PersonalSignEnrollment GetEnrollment(string signId)
        {
            if (_enrollments.TryGetValue(signId, out var e)) return e;

            // 디스크에서 로드 시도
            e = PersonalDataRepository.LoadEnrollment(_profileId, signId);
            if (e == null)
            {
                e = new PersonalSignEnrollment
                {
                    UserProfileId = _profileId,
                    SignId = signId,
                    RequiredAcceptedRepetitions = _settings.EnrollmentRequiredReps,
                    MaxAttemptsPerEnrollmentRound = _settings.EnrollmentMaxAttempts,
                };
            }
            _enrollments[signId] = e;
            return e;
        }

        public bool IsSignReady(string signId)
        {
            return GetEnrollment(signId).IsReady;
        }

        // ─── 등록 시도 ─────────────────────────────────────────────────────

        /// <summary>
        /// 인식 결과를 받아 등록 후보로 평가한다.
        /// 반환: 등록에 수락되었는지.
        /// </summary>
        public EnrollmentAttemptResult TryEnroll(
            string signId,
            SignRecognitionResult recognitionResult,
            List<HandPoseFrame> capturedFrames,
            string sessionId,
            string viewLabel = "front")
        {
            var enrollment = GetEnrollment(signId);
            var result = new EnrollmentAttemptResult { SignId = signId };

            // 이미 완료된 경우
            if (enrollment.IsReady)
            {
                result.Accepted = false;
                result.Reason = "already_enrolled";
                return result;
            }

            // 최대 시도 횟수 초과
            int totalAttempts = enrollment.AcceptedCount +
                (enrollment.RejectedAttemptIds?.Length ?? 0);
            if (totalAttempts >= enrollment.MaxAttemptsPerEnrollmentRound)
            {
                result.Accepted = false;
                result.Reason = "max_attempts_exceeded";
                return result;
            }

            string attemptId = Guid.NewGuid().ToString("N").Substring(0, 12);

            // ── 품질 게이트 ──────────────────────────────────────────────

            // 1. 공용 점수 하한
            if (recognitionResult.GlobalScore < _settings.EnrollmentGlobalScoreMin)
            {
                RejectAttempt(enrollment, attemptId);
                result.Accepted = false;
                result.Reason = "global_score_below_minimum";
                return result;
            }

            // 2. 인식 결과가 passed 또는 높은 점수
            if (recognitionResult.Status != Segmentation.AttemptResultStatus.Passed &&
                recognitionResult.GlobalScore < _settings.PassThreshold)
            {
                RejectAttempt(enrollment, attemptId);
                result.Accepted = false;
                result.Reason = "recognition_not_passed";
                return result;
            }

            // 3. 카메라 품질
            if (recognitionResult.UserFeatures != null &&
                recognitionResult.UserFeatures.HandFrameRatio < 0.5f)
            {
                RejectAttempt(enrollment, attemptId);
                result.Accepted = false;
                result.Reason = "poor_hand_visibility";
                return result;
            }

            // ── 수락 ────────────────────────────────────────────────────

            AcceptAttempt(enrollment, attemptId);

            // 템플릿 저장
            var exemplar = new PersonalExemplar
            {
                AttemptId = attemptId,
                SessionId = sessionId,
                ViewLabel = viewLabel,
                GlobalScore = recognitionResult.GlobalScore,
                Features = recognitionResult.UserFeatures,
                CreatedUtc = DateTime.UtcNow.ToString("o"),
            };

            if (!_exemplars.ContainsKey(signId))
                _exemplars[signId] = new List<PersonalExemplar>();

            var exemplarList = _exemplars[signId];

            // 최대 표본 수 제한
            if (exemplarList.Count >= _settings.MaxPersonalExemplars)
            {
                // 가장 오래된 것 제거
                exemplarList.RemoveAt(0);
            }

            exemplarList.Add(exemplar);

            // 등록 상태 갱신
            if (enrollment.AcceptedCount >= enrollment.RequiredAcceptedRepetitions)
            {
                enrollment.EnrollmentStatus = "ready";
                UpdateEnrollmentQuality(enrollment, exemplarList);
            }
            else
            {
                enrollment.EnrollmentStatus = "in_progress";
            }

            // 저장
            PersonalDataRepository.SaveEnrollment(_profileId, enrollment);

            result.Accepted = true;
            result.AttemptId = attemptId;
            result.AcceptedTotal = enrollment.AcceptedCount;
            result.RequiredTotal = enrollment.RequiredAcceptedRepetitions;
            result.EnrollmentReady = enrollment.IsReady;
            return result;
        }

        // ─── 개인 점수 ─────────────────────────────────────────────────────

        /// <summary>
        /// 개인 템플릿이 있으면 개인 점수를 계산하고 혼합한다.
        /// 공용 절대 하한 미만이면 개인 점수가 높아도 거부.
        /// </summary>
        public float ComputePersonalScore(
            string signId,
            SignFeatureSet userFeatures,
            float globalScore)
        {
            if (!_exemplars.TryGetValue(signId, out var exemplarList) || exemplarList.Count == 0)
                return float.NaN;

            // 개인 점수: 각 템플릿과의 유사도 중 최대값
            float bestPersonal = 0f;
            foreach (var ex in exemplarList)
            {
                if (ex.Features == null || !ex.Features.Valid) continue;
                float sim = CompareFeatures(userFeatures, ex.Features);
                if (sim > bestPersonal) bestPersonal = sim;
            }

            return bestPersonal;
        }

        /// <summary>
        /// 공용+개인 혼합 점수. 공용 하한 검사 포함.
        /// </summary>
        public float ComputeCombinedScore(string signId, float globalScore, float personalScore)
        {
            // 공용 절대 하한 위반 → 개인 점수 무시
            if (globalScore < _settings.GlobalScoreFloor)
                return globalScore;

            if (float.IsNaN(personalScore))
                return globalScore;

            int exemplarCount = _exemplars.TryGetValue(signId, out var list) ? list.Count : 0;
            _settings.GetWeights(exemplarCount, out float gw, out float pw);

            return globalScore * gw + personalScore * pw;
        }

        // ─── 재학습/초기화 ─────────────────────────────────────────────────

        public void ResetEnrollment(string signId)
        {
            if (_enrollments.ContainsKey(signId))
            {
                _enrollments[signId] = new PersonalSignEnrollment
                {
                    UserProfileId = _profileId,
                    SignId = signId,
                    RequiredAcceptedRepetitions = _settings.EnrollmentRequiredReps,
                    MaxAttemptsPerEnrollmentRound = _settings.EnrollmentMaxAttempts,
                };
            }
            if (_exemplars.ContainsKey(signId))
                _exemplars[signId].Clear();

            PersonalDataRepository.DeleteEnrollment(_profileId, signId);
        }

        public void ResetAll()
        {
            _enrollments.Clear();
            _exemplars.Clear();
            PersonalDataRepository.DeleteAllForProfile(_profileId);
        }

        // ─── 내부 ──────────────────────────────────────────────────────────

        void AcceptAttempt(PersonalSignEnrollment enrollment, string attemptId)
        {
            var list = new List<string>(enrollment.AcceptedAttemptIds) { attemptId };
            enrollment.AcceptedAttemptIds = list.ToArray();
        }

        void RejectAttempt(PersonalSignEnrollment enrollment, string attemptId)
        {
            var list = new List<string>(enrollment.RejectedAttemptIds) { attemptId };
            enrollment.RejectedAttemptIds = list.ToArray();
            PersonalDataRepository.SaveEnrollment(_profileId, enrollment);
        }

        void UpdateEnrollmentQuality(PersonalSignEnrollment enrollment, List<PersonalExemplar> exemplars)
        {
            if (exemplars.Count < 2) return;

            // 상호 일관성: 템플릿 간 유사도 평균
            float totalSim = 0f;
            int pairs = 0;
            for (int i = 0; i < exemplars.Count; i++)
            for (int j = i + 1; j < exemplars.Count; j++)
            {
                if (exemplars[i].Features != null && exemplars[j].Features != null)
                {
                    totalSim += CompareFeatures(exemplars[i].Features, exemplars[j].Features);
                    pairs++;
                }
            }

            enrollment.Quality.WithinUserConsistency = pairs > 0 ? totalSim / pairs : 0f;
            enrollment.Quality.GlobalAnchorScore = exemplars[0].GlobalScore;
            enrollment.Quality.Status = enrollment.Quality.WithinUserConsistency >= 0.5f ? "good" : "marginal";
        }

        float CompareFeatures(SignFeatureSet a, SignFeatureSet b)
        {
            if (a == null || b == null || !a.Valid || !b.Valid) return 0f;

            float score = 0f;
            float weight = 0f;

            // 위치 비교
            if (a.Position.HasRight && b.Position.HasRight)
            {
                float dist = Vector2.Distance(a.Position.RightMeanPosition, b.Position.RightMeanPosition);
                score += Mathf.Clamp01(1f - dist / 1.5f);
                weight += 1f;
            }

            // 궤적 비교
            if (a.Trajectory.Valid && b.Trajectory.Valid)
            {
                float dispRatio = b.Trajectory.TotalDisplacement > 0.01f
                    ? a.Trajectory.TotalDisplacement / b.Trajectory.TotalDisplacement
                    : 1f;
                score += Mathf.Clamp01(1f - Mathf.Abs(1f - dispRatio));
                weight += 1f;
            }

            return weight > 0f ? score / weight : 0f;
        }
    }

    /// <summary>
    /// 개인 수어 템플릿.
    /// </summary>
    public sealed class PersonalExemplar
    {
        public string AttemptId;
        public string SessionId;
        public string ViewLabel;
        public float GlobalScore;
        public SignFeatureSet Features;
        public string CreatedUtc;
    }

    /// <summary>
    /// 등록 시도 결과.
    /// </summary>
    public sealed class EnrollmentAttemptResult
    {
        public string SignId;
        public bool Accepted;
        public string AttemptId;
        public string Reason;
        public int AcceptedTotal;
        public int RequiredTotal;
        public bool EnrollmentReady;
    }
}
