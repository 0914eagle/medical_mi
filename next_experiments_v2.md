# 다음 실험 지침서 (v2) — 최신 결과 반영

## 현재까지 결과 요약 (split 적용 후)

```
[성공] Conflict Score 감지:  AUC 0.777 (val) / 0.777 (test)
[성공] Split 재측정 안정:     73.5% (1000) / 72.6% (test 500)
[부분] Context-specific steering: recovery 16.1%, corruption 2.8%
       (전체 wrong_dom: 23.4% / 5.5%)
[약함] 단일 #28696:          0.7% (단독으론 효과 없음)
[실패] MedAbstain 전이:      141개 중 3개만 (2.1%)
```

## 결과가 말하는 방향 전환

```
잘 되는 것: 감지 (detection) — AUC 0.78
안 되는 것: 교정 (steering 천장), 전이 (MedAbstain)

-> 연구 무게중심을 "교정"에서 "감지 + abstain"으로
-> "교정 안 되는 게 오히려 abstain 필요성을 강화"
```

---

# 우선순위별 필요 실험

## 🔴 1순위: 5-fold CV 완성 (지금 fold 0만 있음)

```
문제: 현재 모든 결과가 fold 0 단일 (std=0)
      통계적 신뢰성 없음

해야 할 것:
  conflict score, steering, context-specific 선별을
  fold 0~9 전체로 반복
  -> 평균 +- std 보고

기대:
  AUC 0.78 +- 0.0X
  recovery 1X% +- X%
  -> 이게 논문에 들어갈 정식 수치

우선순위 최상: 지금 수치 전부 "예비"라서
              CV 없이는 보고 못 함
```

---

## 🔴 2순위: MedAbstain 전이 실패 원인 규명

```
관찰: PubMedQA feature -> MedAbstain 적용시 2.1%만 전이
      "메커니즘이 다른가" vs "형식 차이(기술적)인가" 불명

[진단 A] MedAbstain에서 직접 feature 찾기 (전이 말고)
  MedAbstain correct_abstain vs wrong_answered로
  자체 SAE feature 발견 (PubMedQA와 동일 방법)

  비교:
    같은 레이어(18-22)에 신호 있나?
    PubMedQA feature index와 겹치나?

  결과 해석:
    같은 레이어 + 겹침 -> 메커니즘 공유, 전이만 기술적 실패
    다른 레이어 -> 진짜 다른 메커니즘 (의료 context 실패가 단일 아님)

[진단 B] 형식 차이 확인
  MedAbstain은 4지선다(MedQA 기반) + abstain 옵션
  PubMedQA는 yes/no/maybe
  -> 토큰 구조가 달라서 steering vector가 안 맞을 수 있음

  abstain "letter"(E) 처리가 제대로 됐는지
  steering이 logit에 영향 주는지 확인

중요: 이 결과가 "세 데이터셋 공통 메커니즘" 주장의
      성패를 가름
```

---

## 🔴 3순위: 28% 천장 원인 — recovered vs unrecovered (H3)

```
아직 안 한 핵심 분석.
steering으로 교정된 케이스 vs 안 된 케이스 비교.

[이미 있는 데이터 활용]
  steering 결과에서 케이스별 recovered/unrecovered 분류
  (case id 저장되어 있으면 바로)

4가지 비교:
  1. conflict score (이미 계산됨, conflict_score_cv에 case별 있음)
  2. prior 확신도 (conflict_set.json의 prior_probs)
  3. context feature 활성화
  4. SAE reconstruction error

기대:
  recovered가 conflict score 중간, unrecovered가 극단?
  unrecovered가 prior 매우 강함?
  -> "교정 가능/불가능이 본질적으로 다르다" 입증
  -> abstain 대상 = unrecovered 라는 근거

이게 CAI 방법론의 이론적 핵심:
  "감지되지만 교정 안 되는 케이스 = abstain 대상"
```

---

## 🟡 4순위: CAI 통합 파이프라인 평가

```
지금까지는 부분 실험. 이제 통합:

conflict score s 기반 정책:
  s < t_low:      pass (원래 답)
  t_low~t_high:   steer (교정 시도)
  s > t_high:     abstain

[Validation에서] t_low, t_high 결정
[Test에서] 전체 파이프라인 성능:
  - Selective accuracy (abstain 제외 정확도)
  - Coverage (답변한 비율)
  - Safety (wrong 중 abstain 비율)

baseline 비교:
  - No intervention
  - Always steer (SpARE식)
  - CAI (우리)

기대:
  CAI가 selective accuracy 높이고
  위험 케이스 abstain
  -> "감지 강점(AUC 0.78)을 활용한 안전 파이프라인"
```

---

## 🟡 5순위: 일반 domain 데이터셋 추가 (NLP conference용)

```
목적: "의료만"이 아니라 "일반 방법론"으로 positioning
      (NLP conference 타겟)

일반 context-conflict 데이터셋:
  - NQSwap (SpARE가 쓴 것)
  - Macnoise
  - 또는 일반 RAG 데이터셋

확인:
  conflict score / context feature가
  일반 domain에서도 작동하나?
  -> 작동하면 "domain-general 방법"
  -> task: 의료 2-3개 + 일반 2-3개 구성

주의: NQSwap 쓰면 SpARE와 직접 비교 가능
      "우리 CAI vs SpARE" 같은 조건에서
```

---

## 🟢 6순위: Rx-LLM (데이터 도착 후)

```
저자에게 메일 보낸 상태. 데이터 오면:
  PubMedQA와 동일 구조로
  conflict score AUC + steering
  contamination 적어서 가장 깨끗한 검증
```

## 🟢 7순위: Gemma-3-12B 교차 모델

```
전체 파이프라인 Gemma로 재현
-> 모델 agnostic 주장
```

---

# 실행 순서 정리

```
당장 (신뢰성 + 핵심):
  1순위: 5-fold CV 완성 (모든 수치 정식화)
  2순위: MedAbstain 전이 실패 원인 (공통 메커니즘 성패)
  3순위: recovered vs unrecovered (28% 천장 + abstain 근거)

다음 (방법론 완성):
  4순위: CAI 통합 파이프라인
  5순위: 일반 domain 추가 (NLP positioning)

나중 (보강):
  6순위: Rx-LLM
  7순위: Gemma
```

---

# 핵심 메시지 (방향 전환)

```
기존: "feature로 감지하고 steering으로 교정한다"
      -> 교정이 약함 (16-23% 천장)

수정: "feature로 context 무시를 신뢰성 있게 감지하고(AUC 0.78)
       교정 가능한 건 교정, 불가능한 건 abstain한다"
      -> 감지(강점)를 메인, 교정/abstain을 정책으로

이유:
  감지 AUC 0.78 = 확실한 강점
  교정 천장 + 전이 실패 = 약점
  -> 강점 중심으로 재구성
  -> "교정 안 되는 게 abstain 필요성을 증명"
```

---

# 코딩 에이전트 주의사항

```
1. 5-fold CV가 최우선 — 지금 fold 0만이라 std 없음
   모든 핵심 수치(AUC, recovery)를 10-fold로

2. MedAbstain 전이 실패는 두 갈래로 진단:
   - 자체 feature 찾아 레이어 비교 (메커니즘)
   - abstain 토큰/형식 처리 확인 (기술적)

3. recovered/unrecovered는 기존 결과 재활용
   case id가 저장됐는지 먼저 확인
   conflict_score_cv_L20.json에 case별 score 있음

4. CAI 파이프라인은 val에서 threshold, test에서 측정

5. 모든 수치에 평균 +- std (10-fold)
```
