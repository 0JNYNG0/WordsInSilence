using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// HandPoseRecording 시계열 비교 알고리즘 (Phase 6-C).
    ///
    /// 개선 사항:
    ///  1. TrimInactive  — 손이 없는 앞뒤 구간(팔 들어올리기/내리기)을 제거하여
    ///                     실제 동작 구간만 비교한다.
    ///  2. DTW           — 균등 리샘플링 대신 Dynamic Time Warping으로
    ///                     두 시퀀스의 최적 정렬을 찾아 타이밍 차이를 흡수한다.
    ///  3. HandShapeScore — XY만 사용, Y축 반전 버전도 시도해 Unity/Python 좌표계 차이 보정.
    ///  4. WristScore     — Body-relative 좌표 최대 거리 2로 완화, Y반전 버전 시도.
    /// </summary>
    public static class HandPoseEvaluator
    {
        // ─── 결과 구조체 ──────────────────────────────────────────────────────

        public struct MatchResult
        {
            public string MotionId;
            public string DisplayName;
            /// <summary>유사도 0~1 (1이 완전 일치).</summary>
            public float Score;
        }

        /// <summary>
        /// FindBestMatch 에서 이 점수 미만이면 "인식 불가" (빈 MatchResult) 를 반환한다.
        /// Inspector 또는 외부에서 조정 가능.
        /// </summary>
        public static float MinConfidence = 0.50f;

        /// <summary>true 로 설정하면 Compare() 가 레퍼런스별 feature 상세 점수를 로그한다.</summary>
        public static bool DiagnosticMode = false;

        // ─── 공개 API ─────────────────────────────────────────────────────────

        /// <summary>두 HandPoseRecording 의 유사도를 0~1 범위로 반환한다.</summary>
        public static float Compare(HandPoseRecording user, HandPoseRecording reference)
        {
            if (user?.Frames == null || reference?.Frames == null) return 0f;

            var userFrames = TrimInactive(user.Frames);
            var refFrames  = TrimInactive(reference.Frames);

            if (userFrames.Length == 0 || refFrames.Length == 0) return 0f;

            float dtwScore    = CompareDTW(userFrames, refFrames);
            float motionScore = ComputeMotionScore(userFrames, refFrames, reference.MotionId);

            return dtwScore * 0.65f + motionScore * 0.35f;
        }

        /// <summary>
        /// references 중 사용자 녹화와 가장 유사한 항목을 반환한다.
        /// 같은 DisplayName 레퍼런스(원본·fast·slow)는 최고점으로 그룹화한 뒤 비교한다.
        /// 로그 형식: "느린, 정상, 빠른 → DisplayName: 최고점"
        /// </summary>
        public static MatchResult FindBestMatch(HandPoseRecording user, IEnumerable<HandPoseRecording> references)
        {
            // DisplayName → 변형별 점수 (slow/normal/fast)
            var variantScores = new Dictionary<string, int[]>(); // [0]=slow [1]=normal [2]=fast
            var groupBest     = new Dictionary<string, MatchResult>();

            foreach (var reference in references)
            {
                float  score       = Compare(user, reference);
                int    scorePct    = Mathf.RoundToInt(score * 100);
                string displayName = string.IsNullOrEmpty(reference.DisplayName)
                                     ? reference.MotionId
                                     : reference.DisplayName;
                string motionId    = reference.MotionId ?? "";

                int variantIdx = motionId.EndsWith("_slow") ? 0 : 1;

                if (!variantScores.ContainsKey(displayName))
                    variantScores[displayName] = new int[] { -1, -1 };
                variantScores[displayName][variantIdx] = scorePct;

                if (!groupBest.TryGetValue(displayName, out var cur) || score > cur.Score)
                {
                    groupBest[displayName] = new MatchResult
                    {
                        MotionId    = motionId,
                        DisplayName = displayName,
                        Score       = score,
                    };
                }
            }

            // 점수 내림차순 정렬 후 상위 5개만 출력
            var sorted = new System.Collections.Generic.List<(string name, int best)>();
            foreach (var kvp in groupBest)
                sorted.Add((kvp.Key, Mathf.RoundToInt(kvp.Value.Score * 100)));
            sorted.Sort((a, b) => b.best.CompareTo(a.best));

            var sb = new System.Text.StringBuilder("[HPE Top5]\n");
            int count = Mathf.Min(5, sorted.Count);
            for (int i = 0; i < count; i++)
                sb.AppendLine($"  {i + 1}위 {sorted[i].name}: {sorted[i].best}점");
            Debug.Log(sb.ToString());

            // 전체 그룹 중 최고
            var bestMatch = new MatchResult { MotionId = "", DisplayName = "", Score = 0f };
            foreach (var kvp in groupBest)
            {
                if (kvp.Value.Score > bestMatch.Score)
                    bestMatch = kvp.Value;
            }

            // 임계값 미만이면 인식 불가 반환
            if (bestMatch.Score < MinConfidence)
                return new MatchResult { MotionId = "", DisplayName = "", Score = bestMatch.Score };

            // 1·2위 점수 차이가 4점 미만이면 불확실 (오답보다 "모름"이 낫다)
            if (sorted.Count >= 2 && sorted[0].best - sorted[1].best < 4)
                return new MatchResult { MotionId = "", DisplayName = "", Score = bestMatch.Score };

            return bestMatch;
        }

        // ─── 좌표계 보정 헬퍼 ────────────────────────────────────────────────

        /// <summary>
        /// 레퍼런스(Python Y↓) vs 사용자(Unity Y↑) + 웹캠 미러 X반전을 모두 고려해
        /// 4가지 flip 조합 중 가장 가까운 거리를 반환한다.
        /// </summary>
        static float BestVec2Dist(Vector2 user, Vector2 refV)
        {
            return Mathf.Min(Mathf.Min(
                Vector2.Distance(user, refV),
                Vector2.Distance(user, new Vector2( refV.x, -refV.y))),
                Mathf.Min(
                Vector2.Distance(user, new Vector2(-refV.x,  refV.y)),
                Vector2.Distance(user, new Vector2(-refV.x, -refV.y))));
        }

        /// <summary>
        /// 4가지 flip 조합(±X, ±Y) 중 사용자 벡터와의 최대 내적을 반환한다.
        /// Z는 깊이 방향으로 미러링 무관.
        /// </summary>
        static float BestVec3Dot(Vector3 user, Vector3 refV)
        {
            return Mathf.Max(Mathf.Max(
                Vector3.Dot(user, refV),
                Vector3.Dot(user, new Vector3( refV.x, -refV.y, refV.z))),
                Mathf.Max(
                Vector3.Dot(user, new Vector3(-refV.x,  refV.y, refV.z)),
                Vector3.Dot(user, new Vector3(-refV.x, -refV.y, refV.z))));
        }

        // ─── 비활성 프레임 제거 ───────────────────────────────────────────────

        /// <summary>
        /// 손이 감지되지 않는 앞뒤 구간을 제거한다.
        /// 동작 준비(팔 들어올리기)와 종료(팔 내리기) 노이즈를 줄인다.
        /// </summary>
        static HandPoseRecordingFrame[] TrimInactive(HandPoseRecordingFrame[] frames)
        {
            if (frames == null || frames.Length == 0) return frames;

            int start = 0;
            while (start < frames.Length && !HasHand(frames[start]))
                start++;

            int end = frames.Length - 1;
            while (end > start && !HasHand(frames[end]))
                end--;

            if (start > end) return frames;

            int count = end - start + 1;
            var result = new HandPoseRecordingFrame[count];
            Array.Copy(frames, start, result, 0, count);
            return result;
        }

        static bool HasHand(HandPoseRecordingFrame f) =>
            f != null && (f.HasLeftHand || f.HasRightHand);

        // ─── DTW 비교 ────────────────────────────────────────────────────────

        /// <summary>
        /// Dynamic Time Warping으로 두 프레임 시퀀스의 유사도를 구한다.
        /// 타이밍 차이(동작 빠르고 느림)를 자동으로 흡수한다.
        /// Sakoe-Chiba 밴드로 과도한 워핑을 제한한다.
        /// </summary>
        static float CompareDTW(HandPoseRecordingFrame[] user, HandPoseRecordingFrame[] reference)
        {
            int n = user.Length;
            int m = reference.Length;

            // Sakoe-Chiba 밴드: 두 시퀀스 길이 차이의 20% + 최소 3프레임
            int band = Mathf.Max(3, Mathf.RoundToInt(Mathf.Max(n, m) * 0.2f));

            float[,] dtw = new float[n, m];

            // 초기화
            for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                dtw[i, j] = float.MaxValue;

            dtw[0, 0] = FrameDistance(user[0], reference[0]);

            for (int i = 1; i < n; i++)
            {
                if (Mathf.Abs(i - 0) <= band)
                    dtw[i, 0] = dtw[i - 1, 0] + FrameDistance(user[i], reference[0]);
            }
            for (int j = 1; j < m; j++)
            {
                if (Mathf.Abs(0 - j) <= band)
                    dtw[0, j] = dtw[0, j - 1] + FrameDistance(user[0], reference[j]);
            }

            for (int i = 1; i < n; i++)
            for (int j = 1; j < m; j++)
            {
                if (Mathf.Abs(i - j) > band) continue;

                float cost = FrameDistance(user[i], reference[j]);
                float prev = Mathf.Min(dtw[i - 1, j],
                             Mathf.Min(dtw[i, j - 1], dtw[i - 1, j - 1]));
                dtw[i, j] = cost + prev;
            }

            // Open-ended endpoint: 레퍼런스 후미 25%의 어느 지점에서도 종료 허용.
            // → 레퍼런스 영상의 "팔 내리기" 꼬리가 사용자 동작 끝에 강제 매핑되는 왜곡을 방지.
            int tailStart = Mathf.Max(0, Mathf.RoundToInt(m * 0.75f));
            float bestEndDist = float.MaxValue;
            int   bestJ       = m - 1;
            for (int j = tailStart; j < m; j++)
            {
                if (dtw[n - 1, j] < bestEndDist)
                {
                    bestEndDist = dtw[n - 1, j];
                    bestJ       = j;
                }
            }

            if (bestEndDist >= float.MaxValue) return 0f;

            // 최소 경로 길이(대각선)로 정규화 → 점수 분산 확대
            // 기존 (n+m) 정규화보다 약 2배 엄격 → 잘못된 매칭에 더 큰 패널티
            float normalizedDist = bestEndDist / Mathf.Max(n, bestJ + 1);
            return Mathf.Clamp01(1f - normalizedDist);
        }

        /// <summary>두 프레임 간 거리 (0=동일, 1=완전 다름).</summary>
        static float FrameDistance(HandPoseRecordingFrame user, HandPoseRecordingFrame reference)
        {
            return 1f - Mathf.Max(ScoreFrame(user, reference), 0f);
        }

        // ─── 프레임 점수 ─────────────────────────────────────────────────────

        static float ScoreFrame(HandPoseRecordingFrame user, HandPoseRecordingFrame reference)
        {
            if (user == null || reference == null) return -1f;

            float score  = 0f;
            float weight = 0f;

            // ── 손 모양 (합계 최대 50%; 두 손이면 각 25%) ────────────────
            bool uR = user.HasRightHand      && user.RightHandPoints      != null;
            bool rR = reference.HasRightHand && reference.RightHandPoints != null;
            bool uL = user.HasLeftHand       && user.LeftHandPoints       != null;
            bool rL = reference.HasLeftHand  && reference.LeftHandPoints  != null;

            int matchingHands = (uR && rR ? 1 : 0) + (uL && rL ? 1 : 0);
            float handSlot    = matchingHands >= 2 ? 0.25f : 0.5f;

            if (uR && rR)
            {
                score  += HandShapeScore(user.RightHandPoints, reference.RightHandPoints) * handSlot;
                weight += handSlot;
            }
            if (uL && rL)
            {
                score  += HandShapeScore(user.LeftHandPoints, reference.LeftHandPoints) * handSlot;
                weight += handSlot;
            }
            if (matchingHands == 0 && (uR || uL) != (rR || rL))
            {
                // 손 유무 자체가 다를 때 패널티
                score  += 0.1f * 0.5f;
                weight += 0.5f;
            }

            // ── 손목 신체 기준 위치 (50%) ─────────────────────────────────
            if (user.HasValidShoulders && reference.HasValidShoulders)
            {
                float ws = WristPositionScore(
                    user.RightWristBodyRelative,      reference.RightWristBodyRelative,
                    user.LeftWristBodyRelative,       reference.LeftWristBodyRelative);
                score  += ws * 0.5f;
                weight += 0.5f;
            }

            return weight > 0f ? score / weight : -1f;
        }

        // ─── 손 모양 비교 ────────────────────────────────────────────────────

        /// <summary>
        /// 손 21개 점의 모양 유사도 (XYZ 사용).
        /// 정규화 기준: 손목~MCP9 XY 거리 (손 크기 기준 → 카메라 거리 무관).
        /// Z는 동일 기준으로 정규화 후 0.7 가중치 적용 (렌즈 특성 차이 완충).
        /// Y축 반전 버전도 시도해 Unity/Python 좌표계 차이를 보정한다.
        /// </summary>
        static float HandShapeScore(Vector3[] a, Vector3[] b)
        {
            if (a == null || b == null || a.Length < 21 || b.Length < 21) return 0f;

            float aScale = Vector2.Distance(new Vector2(a[0].x, a[0].y), new Vector2(a[9].x, a[9].y));
            float bScale = Vector2.Distance(new Vector2(b[0].x, b[0].y), new Vector2(b[9].x, b[9].y));
            if (aScale < 0.001f || bScale < 0.001f) return 0f;

            const float zWeight = 0.7f;

            float normalDist = 0f;
            float flipDist   = 0f;
            for (int i = 1; i < 21; i++)
            {
                Vector3 aNorm     = new Vector3(
                    (a[i].x - a[0].x) / aScale,
                    (a[i].y - a[0].y) / aScale,
                    (a[i].z - a[0].z) / aScale * zWeight);

                Vector3 bNorm     = new Vector3(
                    (b[i].x - b[0].x) / bScale,
                    (b[i].y - b[0].y) / bScale,
                    (b[i].z - b[0].z) / bScale * zWeight);

                Vector3 bNormFlip = new Vector3(bNorm.x, -bNorm.y, bNorm.z); // Z는 반전 없음

                normalDist += Vector3.Distance(aNorm, bNorm);
                flipDist   += Vector3.Distance(aNorm, bNormFlip);
            }

            float dist = Mathf.Min(normalDist, flipDist);
            // 더 가파른 감소: dist=10이면 score=0 (이전보다 2배 엄격)
            return Mathf.Clamp01(1f - dist / 10f);
        }

        // ─── 손목 위치 비교 ──────────────────────────────────────────────────

        /// <summary>
        /// 어깨 기준 정규화 손목 위치 유사도.
        /// 최대 거리 2로 완화, Y반전 버전도 시도.
        /// </summary>
        static float WristPositionScore(
            Vector2 userRight, Vector2 refRight,
            Vector2 userLeft,  Vector2 refLeft)
        {
            float score  = 0f;
            float weight = 0f;

            if (!float.IsNaN(userRight.x) && !float.IsNaN(refRight.x))
            {
                score  += Mathf.Clamp01(1f - BestVec2Dist(userRight, refRight) / 2f);
                weight += 1f;
            }
            if (!float.IsNaN(userLeft.x) && !float.IsNaN(refLeft.x))
            {
                score  += Mathf.Clamp01(1f - BestVec2Dist(userLeft, refLeft) / 2f);
                weight += 1f;
            }

            return weight > 0f ? score / weight : 0f;
        }

        // ─── 동작 피처 (손가락 개폐 방향 + 손목 이동 범위) ─────────────────────

        /// <summary>
        /// 시계열 전체에 걸친 동작 패턴 유사도.
        /// ApertureTrend  : 손가락 개폐 방향 일치도.
        /// WristTrajectory: 손목 수평 이동 범위(Δx) 유사도.
        /// OscillationCount: 손목 방향 전환 횟수 (반복 동작 구분).
        /// PathClosure    : 시작-끝 변위 비율 (원형 vs 직선 동작 구분).
        /// </summary>
        static float ComputeMotionScore(HandPoseRecordingFrame[] user, HandPoseRecordingFrame[] reference, string refId = "")
        {
            float score = 0f; float weight = 0f;
            bool log = DiagnosticMode;

            // ── 손가락 개폐 방향 ───────────────────────────────────────────────
            float? uApt = ApertureTrend(user);
            float? rApt = ApertureTrend(reference);
            if (uApt.HasValue && rApt.HasValue)
            {
                float diff = Mathf.Abs(uApt.Value - rApt.Value);
                score  += Mathf.Clamp01(1f - diff * 0.75f);
                weight += 1f;
            }

            // ── 손목 수평 이동 범위 ────────────────────────────────────────────
            Vector2? uTraj = WristTrajectoryRange(user);
            Vector2? rTraj = WristTrajectoryRange(reference);
            if (uTraj.HasValue && rTraj.HasValue)
            {
                float xDiff = Mathf.Abs(uTraj.Value.x - rTraj.Value.x);
                float yDiff = Mathf.Abs(uTraj.Value.y - rTraj.Value.y);
                score  += (Mathf.Clamp01(1f - xDiff / 0.8f) + Mathf.Clamp01(1f - yDiff / 1.5f)) * 0.5f;
                weight += 1f;
            }

            // ── X방향 반복 횟수 (방향 전환 수) ────────────────────────────────
            int? uOsc = CountOscillations(user);
            int? rOsc = CountOscillations(reference);
            if (uOsc.HasValue && rOsc.HasValue)
            {
                float oscDiff = Mathf.Abs(uOsc.Value - rOsc.Value);
                score  += Mathf.Clamp01(1f - oscDiff * 0.4f);
                weight += 1f;
            }

            // ── Y방향 반복 횟수 (위아래 탭 구분) ─────────────────────────────
            // 감사합니다: 위아래 2번 탭 → 전환 2~4회 / 죄송하다: 한번 내려옴 → 전환 0~1회
            // 절대 높이와 무관하게 동작 패턴으로 구분
            int? uYOsc = CountYOscillations(user);
            int? rYOsc = CountYOscillations(reference);
            if (uYOsc.HasValue && rYOsc.HasValue)
            {
                float yOscDiff = Mathf.Abs(uYOsc.Value - rYOsc.Value);
                score  += Mathf.Clamp01(1f - yOscDiff * 0.4f);
                weight += 1f;
            }

            // ── 경로 폐합도 (원형 vs 직선) ────────────────────────────────────
            float? uClosure = PathClosure(user);
            float? rClosure = PathClosure(reference);
            if (uClosure.HasValue && rClosure.HasValue)
            {
                float closureDiff = Mathf.Abs(uClosure.Value - rClosure.Value);
                score  += Mathf.Clamp01(1f - closureDiff * 2f);
                weight += 1f;
            }

            // ── Z축 진동 진폭 (원형 동작 깊이 판별) ──────────────────────────
            float? uZAmp = WristZAmplitude(user);
            float? rZAmp = WristZAmplitude(reference);
            if (uZAmp.HasValue && rZAmp.HasValue)
            {
                float zDiff = Mathf.Abs(uZAmp.Value - rZAmp.Value);
                score  += Mathf.Clamp01(1f - zDiff * 4f);
                weight += 1f;
            }

            // ── 동작 중심 높이 (Y-flip 보정 포함) ───────────────────────────
            float? uCenterY = WristCenterY(user);
            float? rCenterY = WristCenterY(reference);
            if (uCenterY.HasValue && rCenterY.HasValue)
            {
                // Python(Y↓)과 Unity(Y↑) 좌표계 부호가 반대 → |u + r| 로 비교
                float yDiff = Mathf.Abs(uCenterY.Value + rCenterY.Value);
                score  += Mathf.Clamp01(1f - yDiff / 2.0f);
                weight += 1f;
            }

            // ── Y 진동 크기 (원형=Y도 흔들림, 수평반복=Y 고정) ───────────────
            float? uYStd = WristYOscillation(user);
            float? rYStd = WristYOscillation(reference);
            if (uYStd.HasValue && rYStd.HasValue)
            {
                float yStdDiff = Mathf.Abs(uYStd.Value - rYStd.Value);
                score  += Mathf.Clamp01(1f - yStdDiff * 5f);
                weight += 1f;
            }

            // ── 손-얼굴 근접도 (코·귀·입) ────────────────────────────────────
            Vector3? uFace = AvgWristFaceDistances(user);
            Vector3? rFace = AvgWristFaceDistances(reference);
            if (uFace.HasValue && rFace.HasValue)
            {
                float noseDiff  = Mathf.Abs(uFace.Value.x - rFace.Value.x);
                float earDiff   = Mathf.Abs(uFace.Value.y - rFace.Value.y);
                float mouthDiff = Mathf.Abs(uFace.Value.z - rFace.Value.z);
                float faceScore = (Mathf.Clamp01(1f - noseDiff  * 1f) +
                                   Mathf.Clamp01(1f - earDiff   * 1f) +
                                   Mathf.Clamp01(1f - mouthDiff * 1f)) / 3f;
                score  += faceScore;
                weight += 1f;
            }

            // ── 검지 평균 방향 벡터 (4방향 flip 보정) ────────────────────────
            Vector3? uFinger = AvgIndexFingerDirection(user);
            Vector3? rFinger = AvgIndexFingerDirection(reference);
            if (uFinger.HasValue && rFinger.HasValue)
            {
                score  += (BestVec3Dot(uFinger.Value, rFinger.Value) + 1f) * 0.5f;
                weight += 1f;
            }

            // ── 동작 수평 중심 위치 (X-flip 보정 포함) ───────────────────────
            float? uCenterX = WristCenterX(user);
            float? rCenterX = WristCenterX(reference);
            if (uCenterX.HasValue && rCenterX.HasValue)
            {
                float xDiff = Mathf.Min(
                    Mathf.Abs(uCenterX.Value - rCenterX.Value),
                    Mathf.Abs(uCenterX.Value + rCenterX.Value));
                score  += Mathf.Clamp01(1f - xDiff / 2.0f);
                weight += 1f;
            }

            // ── 두 손 간 평균 거리 ─────────────────────────────────────────────
            float? uTwoHand = AvgTwoHandDistance(user);
            float? rTwoHand = AvgTwoHandDistance(reference);
            if (uTwoHand.HasValue && rTwoHand.HasValue)
            {
                float distDiff = Mathf.Abs(uTwoHand.Value - rTwoHand.Value);
                score  += Mathf.Clamp01(1f - distDiff);
                weight += 1f;
            }

            // ── 두 손 이동 방향 상관관계 ────────────────────────────────────────
            float? uCorr = TwoHandCorrelation(user);
            float? rCorr = TwoHandCorrelation(reference);
            if (uCorr.HasValue && rCorr.HasValue)
            {
                float corrDiff = Mathf.Abs(uCorr.Value - rCorr.Value);
                score  += Mathf.Clamp01(1f - corrDiff);
                weight += 1f;
            }

            // ── 손바닥 방향 법선 벡터 (4방향 flip 보정) ──────────────────────
            Vector3? uPalm = AvgPalmNormal(user);
            Vector3? rPalm = AvgPalmNormal(reference);
            if (uPalm.HasValue && rPalm.HasValue)
            {
                score  += (BestVec3Dot(uPalm.Value, rPalm.Value) + 1f) * 0.5f;
                weight += 1f;
            }

            // ── 손가락별 펼침 프로파일 (너=검지만 높음, 맥주=전체 균등) ──────
            float[]? uFep = AvgFingerExtensionProfile(user);
            float[]? rFep = AvgFingerExtensionProfile(reference);
            if (uFep != null && rFep != null)
            {
                float totalDiff = 0f;
                for (int i = 0; i < 5; i++)
                    totalDiff += Mathf.Abs(uFep[i] - rFep[i]);
                score  += Mathf.Clamp01(1f - totalDiff * 0.6f);
                weight += 1f;
            }

            float finalScore = weight > 0f ? score / weight : 0.5f;

            if (log)
            {
                var sb = new System.Text.StringBuilder($"[Diag] ref={refId}\n");
                sb.AppendLine($"  ApertureTrend  u={ApertureTrend(user):F2} r={ApertureTrend(reference):F2}");
                sb.AppendLine($"  WristTrajDx    u={WristTrajectoryRange(user)?.x:F2} r={WristTrajectoryRange(reference)?.x:F2}");
                sb.AppendLine($"  WristTrajDy    u={WristTrajectoryRange(user)?.y:F2} r={WristTrajectoryRange(reference)?.y:F2}");
                sb.AppendLine($"  OscCount(X)    u={CountOscillations(user)} r={CountOscillations(reference)}");
                sb.AppendLine($"  OscCount(Y)    u={CountYOscillations(user)} r={CountYOscillations(reference)}");
                sb.AppendLine($"  PathClosure    u={PathClosure(user):F2} r={PathClosure(reference):F2}");
                sb.AppendLine($"  WristZAmp      u={WristZAmplitude(user):F2} r={WristZAmplitude(reference):F2}");
                sb.AppendLine($"  CenterY        u={WristCenterY(user):F2} r={WristCenterY(reference):F2}");
                sb.AppendLine($"  YOscillation   u={WristYOscillation(user):F2} r={WristYOscillation(reference):F2}");
                var uf = AvgWristFaceDistances(user); var rf = AvgWristFaceDistances(reference);
                sb.AppendLine($"  FaceDist(N/E/M) u=({uf?.x:F2},{uf?.y:F2},{uf?.z:F2}) r=({rf?.x:F2},{rf?.y:F2},{rf?.z:F2})");
                var ud = AvgIndexFingerDirection(user); var rd = AvgIndexFingerDirection(reference);
                sb.AppendLine($"  IndexDir       u={ud} r={rd}  dot={( ud.HasValue && rd.HasValue ? Vector3.Dot(ud.Value,rd.Value):0):F2}  bestDot={( ud.HasValue && rd.HasValue ? BestVec3Dot(ud.Value,rd.Value):0):F2}");
                sb.AppendLine($"  CenterX        u={WristCenterX(user):F2} r={WristCenterX(reference):F2}");
                sb.AppendLine($"  TwoHandDist    u={AvgTwoHandDistance(user):F2} r={AvgTwoHandDistance(reference):F2}");
                sb.AppendLine($"  TwoHandCorr    u={TwoHandCorrelation(user):F2} r={TwoHandCorrelation(reference):F2}");
                var up = AvgPalmNormal(user); var rp = AvgPalmNormal(reference);
                sb.AppendLine($"  PalmNormal dot={(up.HasValue && rp.HasValue ? Vector3.Dot(up.Value,rp.Value):0):F2}  bestDot={(up.HasValue && rp.HasValue ? BestVec3Dot(up.Value,rp.Value):0):F2}");
                var ue = AvgFingerExtensionProfile(user); var re = AvgFingerExtensionProfile(reference);
                if (ue != null && re != null)
                    sb.AppendLine($"  FingerExt(T/I/M/R/P) u=[{ue[0]:F2},{ue[1]:F2},{ue[2]:F2},{ue[3]:F2},{ue[4]:F2}] r=[{re[0]:F2},{re[1]:F2},{re[2]:F2},{re[3]:F2},{re[4]:F2}]");
                sb.AppendLine($"  → motionScore={finalScore:F3}");
                Debug.Log(sb.ToString());
            }

            return finalScore;
        }

        /// <summary>
        /// 시퀀스 전체에서 손목 body-relative Y 평균 (동작 높이).
        /// 공부(머리 근처)와 사랑(가슴 앞) 등 위치가 다른 동작을 구분한다.
        /// </summary>
        static float? WristCenterY(HandPoseRecordingFrame[] frames)
        {
            float sum = 0f; int count = 0;
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(w.y)) continue;
                sum += w.y; count++;
            }
            if (count < 4) return null;
            return sum / count;
        }

        /// <summary>
        /// 손목 body-relative Y 표준편차.
        /// 원형 동작(사랑)은 Y도 진동해 값이 크고,
        /// 수평 반복 동작(공부·운동)은 Y가 일정해 값이 작다.
        /// </summary>
        static float? WristYOscillation(HandPoseRecordingFrame[] frames)
        {
            var ys = new List<float>();
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(w.y)) continue;
                ys.Add(w.y);
            }
            if (ys.Count < 8) return null;

            float mean = 0f;
            foreach (var y in ys) mean += y;
            mean /= ys.Count;

            float variance = 0f;
            foreach (var y in ys) variance += (y - mean) * (y - mean);
            return Mathf.Sqrt(variance / ys.Count);
        }

        /// <summary>
        /// 손가락 5개(엄지~소지)의 평균 펼침 비율 배열.
        /// MCP→끝 거리 / 손목→중지MCP 거리로 정규화.
        /// 너: [작음, 큼, 작음, 작음, 작음] / 맥주: [중간, 중간, 중간, 중간, 중간].
        /// </summary>
        static float[]? AvgFingerExtensionProfile(HandPoseRecordingFrame[] frames)
        {
            // MCP 인덱스: 엄지2, 검지5, 중지9, 약지13, 소지17
            // 끝   인덱스: 엄지4, 검지8, 중지12, 약지16, 소지20
            int[] mcps = { 2,  5,  9, 13, 17 };
            int[] tips = { 4,  8, 12, 16, 20 };

            float[] sums  = new float[5];
            int count = 0;

            foreach (var f in frames)
            {
                Vector3[] pts = (f.HasRightHand && f.RightHandPoints?.Length >= 21) ? f.RightHandPoints
                              : (f.HasLeftHand  && f.LeftHandPoints?.Length  >= 21) ? f.LeftHandPoints
                              : null;
                if (pts == null) continue;

                float palmScale = Vector3.Distance(pts[0], pts[9]);
                if (palmScale < 0.001f) continue;

                for (int i = 0; i < 5; i++)
                    sums[i] += Vector3.Distance(pts[mcps[i]], pts[tips[i]]) / palmScale;
                count++;
            }

            if (count < 4) return null;
            for (int i = 0; i < 5; i++) sums[i] /= count;
            return sums;
        }

        /// <summary>
        /// 손바닥 법선 벡터 평균 (XYZ 3D).
        /// pt5(검지MCP)→pt17(소지MCP) × pt0(손목)→pt9(중지MCP) 외적.
        /// 손바닥이 위: +Y, 아래: -Y, 정면: -Z (카메라 향함), 뒤: +Z.
        /// 맥주(위), 인사(정면), 싫다(밀기) 등 구분.
        /// </summary>
        static Vector3? AvgPalmNormal(HandPoseRecordingFrame[] frames)
        {
            Vector3 sum = Vector3.zero; int count = 0;
            foreach (var f in frames)
            {
                Vector3[] pts = (f.HasRightHand && f.RightHandPoints?.Length >= 18) ? f.RightHandPoints
                              : (f.HasLeftHand  && f.LeftHandPoints?.Length  >= 18) ? f.LeftHandPoints
                              : null;
                if (pts == null) continue;

                Vector3 lateral  = pts[17] - pts[5];   // 소지MCP - 검지MCP (가로축)
                Vector3 vertical = pts[9]  - pts[0];   // 중지MCP - 손목 (세로축)
                Vector3 normal   = Vector3.Cross(lateral, vertical);
                if (normal.sqrMagnitude < 1e-8f) continue;
                sum += normal.normalized; count++;
            }
            if (count < 4) return null;
            var avg = sum / count;
            return avg.sqrMagnitude > 1e-6f ? (Vector3?)avg.normalized : null;
        }

        /// <summary>
        /// PosePoints[idx]를 body-relative 좌표(어깨 중심·너비 기준)로 변환한다.
        /// </summary>
        static Vector2 PosePointBodyRel(HandPoseRecordingFrame f, int idx)
        {
            if (f.PosePoints == null || f.PosePoints.Length <= idx) return new Vector2(float.NaN, float.NaN);
            if (f.ShoulderWidth < 0.001f)                            return new Vector2(float.NaN, float.NaN);
            var p = f.PosePoints[idx];
            return new Vector2(
                (p.x - f.ShoulderCenter.x) / f.ShoulderWidth,
                (p.y - f.ShoulderCenter.y) / f.ShoulderWidth);
        }

        /// <summary>
        /// 시퀀스 전체에서 손목→코(x), 손목→귀(y), 손목→입(z) 평균 거리.
        /// BlazePose 인덱스: 0=코, 7=왼귀, 8=오른귀, 9=입좌, 10=입우.
        /// </summary>
        static Vector3? AvgWristFaceDistances(HandPoseRecordingFrame[] frames)
        {
            float sumN = 0f, sumE = 0f, sumM = 0f; int count = 0;
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 wrist = float.IsNaN(f.RightWristBodyRelative.x)
                                ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(wrist.x)) continue;

                Vector2 nose   = PosePointBodyRel(f, 0);
                Vector2 lEar   = PosePointBodyRel(f, 7);
                Vector2 rEar   = PosePointBodyRel(f, 8);
                Vector2 mouthL = PosePointBodyRel(f, 9);
                Vector2 mouthR = PosePointBodyRel(f, 10);
                if (float.IsNaN(nose.x) || float.IsNaN(lEar.x) || float.IsNaN(mouthL.x)) continue;

                float dNose  = Vector2.Distance(wrist, nose);
                float dEar   = Mathf.Min(Vector2.Distance(wrist, lEar), Vector2.Distance(wrist, rEar));
                float dMouth = (Vector2.Distance(wrist, mouthL) + Vector2.Distance(wrist, mouthR)) * 0.5f;

                sumN += dNose; sumE += dEar; sumM += dMouth; count++;
            }
            if (count < 4) return null;
            return new Vector3(sumN / count, sumE / count, sumM / count);
        }

        /// <summary>
        /// 시퀀스 전체에서 검지(pt5→pt8) 평균 방향 단위벡터.
        /// 정면 가리킴(너), 위/아래 가리킴 등을 구분한다.
        /// </summary>
        static Vector3? AvgIndexFingerDirection(HandPoseRecordingFrame[] frames)
        {
            Vector3 sum = Vector3.zero; int count = 0;
            foreach (var f in frames)
            {
                Vector3[] pts = (f.HasRightHand && f.RightHandPoints?.Length >= 9) ? f.RightHandPoints
                              : (f.HasLeftHand  && f.LeftHandPoints?.Length  >= 9) ? f.LeftHandPoints
                              : null;
                if (pts == null) continue;
                Vector3 dir = pts[8] - pts[5];
                if (dir.sqrMagnitude < 1e-6f) continue;
                sum += dir.normalized; count++;
            }
            if (count < 4) return null;
            var avg = sum / count;
            return avg.sqrMagnitude > 1e-6f ? (Vector3?)avg.normalized : null;
        }

        /// <summary>
        /// 시퀀스 전체에서 손목 body-relative X 평균 (동작 수평 중심).
        /// 너(오른쪽), 나(왼쪽), 중앙 동작 구분.
        /// </summary>
        static float? WristCenterX(HandPoseRecordingFrame[] frames)
        {
            float sum = 0f; int count = 0;
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(w.x)) continue;
                sum += w.x; count++;
            }
            if (count < 4) return null;
            return sum / count;
        }

        /// <summary>
        /// 두 손이 모두 감지된 프레임에서의 손목 간 평균 거리.
        /// 손이 붙어있는 동작 vs 벌어진 동작 구분.
        /// </summary>
        static float? AvgTwoHandDistance(HandPoseRecordingFrame[] frames)
        {
            float sum = 0f; int count = 0;
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders || !f.HasLeftHand || !f.HasRightHand) continue;
                if (float.IsNaN(f.LeftWristBodyRelative.x) || float.IsNaN(f.RightWristBodyRelative.x)) continue;
                sum += Vector2.Distance(f.LeftWristBodyRelative, f.RightWristBodyRelative);
                count++;
            }
            if (count < 4) return null;
            return sum / count;
        }

        /// <summary>
        /// 두 손이 같은 방향으로 움직이는지(+1) 반대 방향인지(-1) 측정.
        /// X축 이동 방향의 프레임 간 부호 일치율.
        /// </summary>
        static float? TwoHandCorrelation(HandPoseRecordingFrame[] frames)
        {
            float dotSum = 0f; int count = 0;
            for (int i = 1; i < frames.Length; i++)
            {
                var prev = frames[i - 1]; var curr = frames[i];
                if (!prev.HasLeftHand || !curr.HasLeftHand)   continue;
                if (!prev.HasRightHand || !curr.HasRightHand) continue;
                if (float.IsNaN(prev.LeftWristBodyRelative.x) || float.IsNaN(prev.RightWristBodyRelative.x)) continue;

                float lDx = curr.LeftWristBodyRelative.x  - prev.LeftWristBodyRelative.x;
                float rDx = curr.RightWristBodyRelative.x - prev.RightWristBodyRelative.x;
                if (Mathf.Abs(lDx) + Mathf.Abs(rDx) < 0.001f) continue;

                dotSum += (lDx * rDx > 0f) ? 1f : -1f;
                count++;
            }
            if (count < 4) return null;
            return dotSum / count;
        }

        /// <summary>
        /// Pose 손목 Z좌표의 진동 진폭 (max - min).
        /// 원형 동작(사랑)은 앞뒤로 움직여 진폭이 크고,
        /// 평면 반복 동작(공부·운동)은 진폭이 작다.
        /// </summary>
        static float? WristZAmplitude(HandPoseRecordingFrame[] frames)
        {
            const int RIGHT_WRIST = 16;
            const int LEFT_WRIST  = 15;

            float min = float.MaxValue, max = float.MinValue;
            int count = 0;

            foreach (var f in frames)
            {
                if (f.PosePoints == null || f.PosePoints.Length <= RIGHT_WRIST) continue;

                float z = f.PosePoints[RIGHT_WRIST].z;
                if (float.IsNaN(z)) z = f.PosePoints[LEFT_WRIST].z;
                if (float.IsNaN(z)) continue;

                if (z < min) min = z;
                if (z > max) max = z;
                count++;
            }

            if (count < 8) return null;
            return max - min;
        }

        /// <summary>
        /// 손목 Y 이동에서 방향 전환 횟수를 센다.
        /// 위아래 탭 동작(감사합니다=2~4회)과 단순 아래 스윕(죄송하다=0~1회)을 구분한다.
        /// 절대 높이와 무관하므로 카메라 위치·체형 차이에 강건하다.
        /// </summary>
        static int? CountYOscillations(HandPoseRecordingFrame[] frames)
        {
            var ys = new List<float>();
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(w.y)) continue;
                ys.Add(w.y);
            }
            if (ys.Count < 8) return null;

            var smoothed = new float[ys.Count];
            int win = 2;
            for (int i = 0; i < ys.Count; i++)
            {
                int s = Mathf.Max(0, i - win);
                int e = Mathf.Min(ys.Count - 1, i + win);
                float sum = 0f;
                for (int j = s; j <= e; j++) sum += ys[j];
                smoothed[i] = sum / (e - s + 1);
            }

            const float minChange = 0.04f;
            int reversals = 0;
            int dir = 0;
            for (int i = 1; i < smoothed.Length; i++)
            {
                float d = smoothed[i] - smoothed[i - 1];
                int newDir = d > minChange ? 1 : (d < -minChange ? -1 : 0);
                if (newDir != 0 && newDir != dir)
                {
                    if (dir != 0) reversals++;
                    dir = newDir;
                }
            }
            return reversals;
        }

        /// <summary>
        /// 손목 X 이동에서 방향 전환 횟수를 센다.
        /// 반복 동작(공부=2~3회)과 단순 동작(자다=0~1회)을 구분한다.
        /// </summary>
        static int? CountOscillations(HandPoseRecordingFrame[] frames)
        {
            var xs = new List<float>();
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(w.x)) continue;
                xs.Add(w.x);
            }
            if (xs.Count < 8) return null;

            // 이동평균 스무딩 (윈도우 5)
            var smoothed = new float[xs.Count];
            int win = 2;
            for (int i = 0; i < xs.Count; i++)
            {
                int s = Mathf.Max(0, i - win);
                int e = Mathf.Min(xs.Count - 1, i + win);
                float sum = 0f;
                for (int j = s; j <= e; j++) sum += xs[j];
                smoothed[i] = sum / (e - s + 1);
            }

            // 방향 전환 카운트 (어깨너비의 4% 이상 움직여야 유효)
            const float minChange = 0.04f;
            int reversals = 0;
            int dir = 0;
            for (int i = 1; i < smoothed.Length; i++)
            {
                float d = smoothed[i] - smoothed[i - 1];
                int newDir = d > minChange ? 1 : (d < -minChange ? -1 : 0);
                if (newDir != 0 && newDir != dir)
                {
                    if (dir != 0) reversals++;
                    dir = newDir;
                }
            }
            return reversals;
        }

        /// <summary>
        /// 경로 폐합도: 총 이동 거리 대비 시작→끝 직선 거리 비율.
        /// 0 = 완전 원형 (제자리), 1 = 완전 직선.
        /// 원형 동작(사랑)과 직선 동작(자다)을 구분한다.
        /// </summary>
        static float? PathClosure(HandPoseRecordingFrame[] frames)
        {
            var positions = new List<Vector2>();
            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative : f.RightWristBodyRelative;
                if (float.IsNaN(w.x)) continue;
                positions.Add(w);
            }
            if (positions.Count < 8) return null;

            float pathLength = 0f;
            for (int i = 1; i < positions.Count; i++)
                pathLength += Vector2.Distance(positions[i], positions[i - 1]);

            if (pathLength < 0.01f) return 0f;

            float netDisplacement = Vector2.Distance(positions[0], positions[positions.Count - 1]);
            return Mathf.Clamp01(netDisplacement / pathLength);
        }

        /// <summary>
        /// 시퀀스 전체에서 손목 body-relative 이동 범위 (Δx, Δy).
        /// 이동이 클수록 큰 값 → 동선이 긴 동작과 정적인 동작 구분.
        /// </summary>
        static Vector2? WristTrajectoryRange(HandPoseRecordingFrame[] frames)
        {
            float minX =  float.MaxValue, maxX = float.MinValue;
            float minY =  float.MaxValue, maxY = float.MinValue;
            int count = 0;

            foreach (var f in frames)
            {
                if (!f.HasValidShoulders) continue;
                Vector2 w = float.IsNaN(f.RightWristBodyRelative.x)
                            ? f.LeftWristBodyRelative
                            : f.RightWristBodyRelative;
                if (float.IsNaN(w.x)) continue;

                if (w.x < minX) minX = w.x;
                if (w.x > maxX) maxX = w.x;
                if (w.y < minY) minY = w.y;
                if (w.y > maxY) maxY = w.y;
                count++;
            }

            if (count < 4) return null;
            return new Vector2(maxX - minX, maxY - minY);
        }

        // ─── 공개 유틸 (SVMClassifier 사용) ──────────────────────────────────────

        /// <summary>비활성(손 없음) 앞뒤 구간을 제거한 배열을 반환한다 (SVMClassifier용 공개 래퍼).</summary>
        public static HandPoseRecordingFrame[] TrimInactivePublic(HandPoseRecordingFrame[] frames)
            => TrimInactive(frames);

        /// <summary>
        /// 26개 float 피처 벡터를 반환한다. (SVMClassifier 학습/추론용)
        /// 계산 불가 피처는 float.NaN.
        /// 피처 순서: ApertureTrend, WristTrajDx, WristTrajDy, OscCountX, OscCountY,
        /// PathClosure, WristZAmp, WristCenterY, WristYOscillation,
        /// FaceDistNose, FaceDistEar, FaceDistMouth,
        /// IndexDirX, IndexDirY, IndexDirZ,
        /// WristCenterX, TwoHandDist, TwoHandCorr,
        /// PalmNormalX, PalmNormalY, PalmNormalZ,
        /// FingerExtThumb..Pinky (5개).
        /// </summary>
        public static float[] ExtractFeatureVector(HandPoseRecordingFrame[] frames)
        {
            var f = new float[26];
            int i = 0;

            f[i++] = ApertureTrend(frames) ?? float.NaN;

            var traj = WristTrajectoryRange(frames);
            f[i++] = traj.HasValue ? traj.Value.x : float.NaN;
            f[i++] = traj.HasValue ? traj.Value.y : float.NaN;

            var oscX = CountOscillations(frames);
            f[i++] = oscX.HasValue ? (float)oscX.Value : float.NaN;

            var oscY = CountYOscillations(frames);
            f[i++] = oscY.HasValue ? (float)oscY.Value : float.NaN;

            f[i++] = PathClosure(frames)      ?? float.NaN;
            f[i++] = WristZAmplitude(frames)  ?? float.NaN;
            f[i++] = WristCenterY(frames)     ?? float.NaN;
            f[i++] = WristYOscillation(frames) ?? float.NaN;

            var face = AvgWristFaceDistances(frames);
            f[i++] = face.HasValue ? face.Value.x : float.NaN;
            f[i++] = face.HasValue ? face.Value.y : float.NaN;
            f[i++] = face.HasValue ? face.Value.z : float.NaN;

            var idir = AvgIndexFingerDirection(frames);
            f[i++] = idir.HasValue ? idir.Value.x : float.NaN;
            f[i++] = idir.HasValue ? idir.Value.y : float.NaN;
            f[i++] = idir.HasValue ? idir.Value.z : float.NaN;

            f[i++] = WristCenterX(frames)        ?? float.NaN;
            f[i++] = AvgTwoHandDistance(frames)  ?? float.NaN;
            f[i++] = TwoHandCorrelation(frames)  ?? float.NaN;

            var palm = AvgPalmNormal(frames);
            f[i++] = palm.HasValue ? palm.Value.x : float.NaN;
            f[i++] = palm.HasValue ? palm.Value.y : float.NaN;
            f[i++] = palm.HasValue ? palm.Value.z : float.NaN;

            var fep = AvgFingerExtensionProfile(frames);
            f[i++] = fep != null ? fep[0] : float.NaN;
            f[i++] = fep != null ? fep[1] : float.NaN;
            f[i++] = fep != null ? fep[2] : float.NaN;
            f[i++] = fep != null ? fep[3] : float.NaN;
            f[i++] = fep != null ? fep[4] : float.NaN;

            return f;
        }

        /// <summary>
        /// 시퀀스 전반부 대비 후반부에서 손가락 끝–손목 평균 거리 변화 방향.
        /// 양수 = 열림(extension), 음수 = 닫힘(flexion). 범위 [-1, 1].
        /// </summary>
        static float? ApertureTrend(HandPoseRecordingFrame[] frames)
        {
            var apts = new List<float>();
            foreach (var f in frames)
            {
                Vector3[] pts = null;
                if (f.HasRightHand && f.RightHandPoints != null && f.RightHandPoints.Length >= 21)
                    pts = f.RightHandPoints;
                else if (f.HasLeftHand && f.LeftHandPoints != null && f.LeftHandPoints.Length >= 21)
                    pts = f.LeftHandPoints;
                if (pts == null) continue;

                // 손목(0)에서 각 손가락 끝(4,8,12,16,20)까지 XY 거리 평균
                float apt = 0f;
                apt += Vector2.Distance(new Vector2(pts[0].x, pts[0].y), new Vector2(pts[4].x,  pts[4].y));
                apt += Vector2.Distance(new Vector2(pts[0].x, pts[0].y), new Vector2(pts[8].x,  pts[8].y));
                apt += Vector2.Distance(new Vector2(pts[0].x, pts[0].y), new Vector2(pts[12].x, pts[12].y));
                apt += Vector2.Distance(new Vector2(pts[0].x, pts[0].y), new Vector2(pts[16].x, pts[16].y));
                apt += Vector2.Distance(new Vector2(pts[0].x, pts[0].y), new Vector2(pts[20].x, pts[20].y));
                apts.Add(apt / 5f);
            }

            if (apts.Count < 4) return null;

            // 전반부 평균 vs 후반부 평균
            int   mid    = apts.Count / 2;
            float first  = 0f;
            float second = 0f;
            for (int i = 0;   i < mid;        i++) first  += apts[i];
            for (int i = mid; i < apts.Count; i++) second += apts[i];
            first  /= mid;
            second /= (apts.Count - mid);

            // 전체 변화 범위로 정규화
            float min = apts[0], max = apts[0];
            foreach (var v in apts) { if (v < min) min = v; if (v > max) max = v; }
            float range = max - min;
            if (range < 0.001f) return 0f;

            return Mathf.Clamp((second - first) / range, -1f, 1f);
        }

    }
}
