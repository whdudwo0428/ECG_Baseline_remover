# How to Use

간단 사용 가이드 (패치 반영 버전 기준)

## 1) calibration.py 뷰어 실행
- 명령: `python calibration.py`
- 기능:
  - 하이브리드 베이스라인 제거(Hybrid BL++) + 선택적 Residual Refit
  - QRS-aware, 변화점 보호, 고분산(HV) 마스크 등
  - 패치 내용: 노이즈/기울기↑ → λ(스무딩 강도)↑로 반전. 과적응 억제
  - 콘솔 출력: DriftMetrics BEFORE(original)/AFTER(fixed)
    - `std_baseline`(낮을수록 기준선이 덜 흔들림)
    - `mad_from_zero`(낮을수록 0선 중심 정렬이 잘됨)
    - `var_grad_baseline`(낮을수록 기준선 출렁임 적음)
    - `lf_power_frac_baseline`(≤0.5Hz 저주파 비율, 낮을수록 드리프트 억제)
- 확인 포인트:
  - Baseline/Corrected 표시로 기준선 평탄화 확인
  - DriftMetrics BEFORE/AFTER 값 비교(낮을수록 개선)

## 2) vis.py 시각화/메트릭
- 버전 선택 실행:
  - 원본(v0): `python vis.py --version 0`
  - 패치(v1): `python vis.py --version 1`
  - 최적화(v2): `python vis.py --version 2`
  - 둘 다 비교(v0,v1): `python vis.py --version both --fft-compare --fft-log`
  - 모두 비교(v0,v1,v2): `python vis.py --version all --fft-compare --fft-log`
- 기본 옵션:
  - `--file <path>`: 입력 JSON 경로(기본: repo 내 샘플)
  - `--fs <Hz>`: 샘플링 레이트(기본 250)
  - `--fft-compare`: 밴드별 Raw vs Corrected 비교 플롯 추가 저장
  - `--fft-log`: FFT 진폭 dB 스케일(20·log10) 표시
- 출력 디렉토리:
  - v0: `vis/version0/`
  - v1: `vis/version1/`
  - v2: `vis/version2/`
  - both: `vis/version0/`와 `vis/version1/` 모두 생성(각 버전별 시각화 저장)
  - all: `vis/version0/`, `vis/version1/`, `vis/version2/` 모두 생성

### 생성 결과(버전 디렉토리 하위에 저장)
1) 타임 도메인
   - `time_raw-baseline-corrected_{base}.png`
   - Raw/Baseline/Corrected 3단 플롯로 0선(기준선) 평탄화 및 대칭성 확인

2) FFT (밴드별, 총 8개 + 비교 옵션)
   - 밴드: noise(0–0.5Hz), low(0.5–5Hz), mid(5–15Hz), high(15–40Hz)
   - Raw 단일: `fft_raw_{band}_{base}.png`
   - Corrected 단일: `fft_corrected_{band}_{base}.png`
   - 옵션 `--fft-compare` 시 비교 플롯: `fft_compare_{band}_{base}.png`
   - 목적: 저주파(0–0.5Hz) 성분 감소로 드리프트 억제 여부, ECG 관심대역 변화 확인

3) PSD (Welch)
   - `psd_corrected_{base}.png`
   - 보정 신호의 잔류 저주파 파워 확인(미세 드리프트 탐지)

4) Residual 히스토그램
   - `hist_residuals_{base}.png`
   - `y_raw - baseline` 분포가 0 중심 대칭인지로 오프셋/바이어스 유무 확인

5) Baseline Rolling Std
   - `rollingstd_baseline_{base}.png`
   - 구간별 기준선 불안정(출렁임) 구간 식별

6) 메트릭 CSV (세그먼트별, 덮어쓰기 — 버전 공통)
   - 세그먼트 범위 고정: 1000s → 3500s, 500s 간격(총 5개)
   - 버전별 롱 포맷(공통 경로): `vis/metrics_segments_{base}.csv`
     - 열: `version, seg_idx, t_start, t_end, std_baseline, mad_from_zero, var_grad_baseline, lf_power_frac_baseline`
   - 버전쌍 비교(공통 경로): `vis/metrics_segments_compare_{va}_{vb}_{base}.csv` (예: v0_v1, v0_v2, v1_v2)
     - 행 우선(트랜스포즈) 구성 예시:
       - `seg_idx | 0 | 1 | 2 | 3 | 4`
       - `t_start | 1000 | 1500 | 2000 | 2500 | 3000`
       - `t_end   | 1500 | 2000 | 2500 | 3000 | 3500`
       - `<메트릭>_{va}` 행들, `<메트릭>_{vb}` 행들(예: std_baseline_v0, std_baseline_v1, ...)
       - 구분선 한 줄
       - `<메트릭>_improve_pct` 행들(낮을수록 개선)
   - 기준: 값이 낮을수록 안정(개선율=(v0−v1)/|v0|*100)
   - 저장 위치: 공통 디렉토리 `vis/` (버전 폴더 밖) — 한눈에 버전 비교 용이

## 3) 빠른 체크리스트
- 드리프트가 심한 데이터에서:
  - `python vis.py --version both --fft-compare --fft-log`
  - v1이 v0 대비 저주파(0–0.5Hz) FFT/PSD가 감소하고, CSV의 메트릭이 전반적으로 낮아지는지 확인
  - time-domain 플롯에서 기준선이 평탄(0선 유지)이고 보정 신호가 0선 대칭으로 진동하는지 확인
