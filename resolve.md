# ECG 데이터 정규화·보정 보고서 (초안, md)

## 0. 결론

* **3단계 파이프라인**으로 문제를 순차 해결합니다:
  **① 페이싱/스파이크 아티팩트 제거 → ② 베이스라인 드리프트 제거 + 등전위선(isoelectric) 복원(QVRi) → ③ PQ-세그먼트 군집 기반 등전위편향 보정·검증.**
* **평가 지표**는 *Fasano & Villani(2015)*의 재구성오차(정규화 L2)와 *Le et al.(2019)*의 PQ 기반 검증(PD-A/B, |bias−median(PQ)|의 μ/σ)을 핵심으로 삼고, 내부 품질지표(std_baseline, mad_from_zero 등)로 운영 모니터링을 병행합니다.  ([eurasip.org][1])
* 기대효과: **0선 기준 대칭 진동**으로 표준화, **PQRST 보존**, **급격 변화(예: 865–875 s)에서 파형 붕괴 방지**. 우리 데이터의 문제·목표 정의는 `problem.md`에 정리된 바와 같습니다.  

---

## 1. 데이터 문제와 목표

### 1.1 문제 요약

* **Baseline wander**: 저주파 요인(호흡·전극 이동 등)로 0선이 흔들려 **기준축 불안정**.
* **Spike/페이싱 유사 아티팩트**: 급격 진폭 이탈로 **보정 신호가 붕괴**, PQRST가 손실(865–875 s 구간에서 두드러짐). 

### 1.2 목표

* **0선 고정/대칭 진동**(정규화), **스파이크 제거**, **PQRST 보존**, 최종적으로 **ECG 품질 향상**. 

---

## 2. 참고 연구와 파이프라인 설계

### 단계 ① 스파이크(페이싱) 아티팩트 제거 → 파형 보전

* **근거 연구**:

  * *Harvey et al., 2020 (IEEE SPMB)*: **수정 Z-점수(modified Z-score)**로 급격 이상치를 검출하고, 제거 구간을 **쌍곡코사인 보간**으로 메우는 방법을 제시. QRS 영역 측정 왜곡(면적) 개선을 보고. 
  * 추가로 2021–2023의 페이싱 아티팩트 제거 연구들(반자동·프레임워크 확장)도 유효성 뒷받침. ([PubMed][2])
* **적용 지침(우리 파이프라인)**

  * 1. 1차 detrend 후 1차 차분·중앙절대편차 기반 **modified Z**를 계산 → 동적 임계값으로 스파이크 마스킹.
  * 2. 마스크 구간을 **cosh 보간**(양 끝 접선연속)으로 대체 → 파형 자연스러움 유지.
  * 3. QRS 검출 보호 마스크로 **피크 보존**(QRS 내부 스파이크는 치환 길이·형상을 더 보수적으로).
  * 4. 이후 단계의 베이스라인 추정에 영향 최소화를 위해 **먼저 수행**.

### 단계 ② 베이스라인 드리프트 제거 + **등전위선 복원(QVRi)**

* **근거 연구**: *Fasano & Villani, 2015 (CinC)*. **Quadratic Variation Reduction(QVR)**에 **소수의 결절점(knots)** 제약을 추가해 등전위 레벨(0 V) 복원을 함께 달성(QVRi). **비볼록 아님/전역해 보장(볼록문제)**, **O(n)** 복잡도, **CSI 대비 ST 보존 우수**. 
* **핵심 아이디어**

  * “측정 ECG = 실제 ECG + 저변동(저주파) 성분(베이스라인)” 가정.
  * **변동도(Quadratic Variation)** 최소·상한 제약과 **소수의 등전위 결절점**(예: PQ 중간점, TP 중간점 일부) 통과 제약으로 베이스라인을 추정 → detrend. 
* **적용 지침(우리 파이프라인)**

  * 1. **0.5–0.67 Hz HPF**로 1차 안정화(위상영→ 왜곡 최소) 후, ([eurasip.org][1])
  * 2. **QVRi**로 베이스라인 재추정·제거 + **등전위선 복원**(결절점은 신뢰 PQ/TP 일부만 사용; beat마다 필요 없음). 

### 단계 ③ **PQ-세그먼트 군집 기반 등전위편향 보정·검증**

* **근거 연구**: *Le et al., 2019 (EUSIPCO)*. 베이스라인 제거 후에도 **등전위 보정은 별개 문제**일 수 있음을 지적. **PQ 구간에서 등전위편향(0 V 대비 오프셋)**을 군집으로 추정해 **PD-A/B, |bias−median(PQ)|의 μ/σ** 지표로 검증. **R-left**가 단순·효율·성능 우수. ([eurasip.org][1])
* **적용 지침**

  * 1. R-피크 기준 **PQ 세그먼트 샘플 집합**을 구성하고 정렬·특징화.
  * 2. **주 군집의 중심 = isoelectric bias(ˆb)**로 정의 → 전 구간에 **offset 보정**.
  * 3. **검증**: (A) 위치가 PQ 내인가(PD-A), (B) |median(PQ)−ˆb|<σ_th (PD-B), (C) |median(PQ)−ˆb|의 μ/σ. ([eurasip.org][1])

---

## 3. 우리 코드와의 정합(파일·함수 매핑)

* **1차 HPF(0.5–0.67 Hz)**: `remove_baseline_drift(y, fs, cutoff=0.5, order=4)`는 **양방향(zero-phase)**로 0선 중심화 수행(운영상 안전한 초기 안정화). 
* **내부 품질지표(운영 모니터링)**: `compute_drift_metric(y_corr, b_final, fs)`는 **std_baseline, mad_from_zero, ∇b 분산, LF 파워비**를 제공 → 단계별 개선량 로그에 사용.  
* **표시(UI)**: 가공(보정) 신호의 **현재 기준축(= 등전위선)**은 **QVRi+군집 보정 후의 오프셋 누적치**를 별도 곡선으로 그려 실제 축 변화를 시각화(“주황 점선”은 ‘추정 축’을 보여야 함).
* 현재 코드인 calibration.py의 모든 함수별 구동시간 시각화, ui 등 전체적인 기능적인 코드는 유지한 채 사용하지 않는 기능이나 쓸모없는 코드 정리 및 최적화
* **UI 요구:** 단계별 처리 버튼 3개 제공 — `[s1 스파이크 제거]`, `[s2 QVRi]`, `[s3 PQ-군집]`.
* **동작 규칙:** `s1` 클릭 시 **1단계만** 적용된 보정파형을 즉시 갱신, `s2` 클릭 시 **1+2단계 누적** 결과를, `s3` 클릭 시 **1+2+3단계 누적** 결과를 출력(플롯·메트릭 동기 갱신).
* 해당 처리 버튼을 누를 때 생성 파형에 해당하는 baseline도 자동으로 함께 출력되도록 (현재 "가공 baseline 출력" 버튼을 없애고 동시에 세 버튼과 기능 통합되도록 구현)


---

## 4. 평가 지표(메트릭) 설계

### 4.1 핵심 외부 지표

* **정규화 재구성오차 ε** (*Fasano & Villani 2015*):
  [
  \varepsilon = \frac{\lVert z_{\text{out}} - z_0 \rVert_2}{\lVert z_0 \rVert_2}
  ]
  (합성 드리프트/레벨시프트 주입 실험으로 z₀ 대비 평가; **에러 EDF**로 방법 비교). 
* **PQ 기반 등전위 검증** (*Le et al. 2019*):
  **PD-A**(PQ 내 위치 비율), **PD-B**(|median(PQ)−ˆb|<σ_th 비율), **μ/σ**(절대차 통계). 최선 사례로 PD-A≈99.6%, PD-B≈98.4%, μ≈6.3 µV, σ≈12.0 µV. ([eurasip.org][1])

### 4.2 내부 운영 지표(로그/대시보드)

* **std_baseline**, **mad_from_zero**, **var(∇baseline)**, **LF power frac(≤0.5 Hz)** — 단계별 전/후를 표준 보고. 

---

## 5. 실험 설계(권장)

| 단계        | 입력       | 처리                      | 출력/로그                      | 1차 합격 기준                                                        |
| --------- | -------- | ----------------------- | -------------------------- | --------------------------------------------------------------- |
| ① 스파이크 제거 | HPF 후 신호 | modified-Z 검출 → cosh 보간 | 스파이크 마스크 길이·개수, QRS 보호율    | QRS 인접 왜곡 0에 가깝게, QRS 면적 오차↓(Harvey 방식 준거)                      |
| ② QVRi    | ①출력      | QVR(λ) + 소수 결절점         | detrended, baseline, knots | 합성실험 ε 중간값↓, ST 유지                                              |
| ③ PQ-군집   | ②출력      | R-left 군집→offset 보정     | ˆb, PD-A/B, μ/σ            | PD-A≥98%, PD-B≥95%, μ<15 µV, σ<25 µV (초기 기준) ([eurasip.org][1]) |

---

## 6. 리스크·대응

* **P/T 경계 주석 부정확** → PQ 추정 오류: PQ 후보를 다중 세그먼트로 두고 **강건 통계(중앙값·IQR)** + 이상치 제거. ([eurasip.org][1])
* **페이싱/스파이크가 QRS 내부에 존재**: 보간 길이·형상을 보수적으로, QRS 경계 접선 연속성 제약. 
* **실데이터는 z₀ 부재**: ε 평가는 **합성 주입 테스트**로 수행하고, 실데이터는 **PQ 지표 + 내부 품질지표**로 대체.

---

## 7. 문서·결과 정리 가이드

* 각 단계별로 **Before/After 플롯**, **지표 테이블** 기록.
- `problem.md`의 문제정의·목표와 연결해 개선 효과를 명시. 
* 코드 주석에는 각 함수 상단에 **근거 논문·파라미터 근거**를 링크(예: HPF 컷오프 0.67 Hz 사용 사유). ([eurasip.org][1])

---

## 8. 참고 문헌

1. **ECG Baseline Wander Removal with Recovery of the Isoelectric Level**, A. Fasano, V. Villani, CinC 2015. 핵심: QVRi(결절점 제약)로 등전위선 복원·ST 보존·O(n). 지표: 정규화 L2 재구성오차·EDF. 
2. **Validation of Baseline Wander Removal and Isoelectric Correction in ECGs Using Clustering**, K. Le et al., EUSIPCO 2019. 핵심: **PQ 군집**으로 등전위편향 추정·검증(PD-A/B, μ/σ); 4차 BW HPF=0.67 Hz 전처리. ([eurasip.org][1])
3. **Automated Pacing Artifact Removal in Electrocardiograms**, C. Harvey et al., IEEE SPMB 2020. 핵심: **modified Z-score** 기반 스파이크 검출 + **쌍곡코사인 보간**으로 QRS 측정 개선. 
   (보강) 페이싱 아티팩트 제거 관련 후속/유관 문헌도 참고. ([PubMed][2])

---

## 부록 A. 구현 체크리스트(요약)

* [ ] 단계①: modified-Z 스파이크 마스크 + cosh 보간(피크 보호)
* [ ] 단계②: HPF(0.5–0.67 Hz) → QVRi(결절점: PQ/TP 일부)
* [ ] 단계③: PQ-군집 오프셋 보정 + PD-A/B, μ/σ 계산
* [ ] 로그: `compute_drift_metric` 전/후 비교, 실패 구간 스냅샷 저장.

[1]: https://www.eurasip.org/Proceedings/Eusipco/eusipco2019/Proceedings/papers/1570529931.pdf "Validation of Baseline Wander Removal and Isoelectric Correction in Electrocardiograms Using Clustering"
[2]: https://pubmed.ncbi.nlm.nih.gov/33872969/?utm_source=chatgpt.com "Detection and removal of pacing artifacts prior to automated ..."
