# Baseline Drift + Spike Artifact — Problem & Approach

## 문제
- ECG(심전도) 신호 처리에서 발생하는 베이스라인 드리프트(baseline wander)와 스파이크 아티팩트(spike artifact)가 주요 문제입니다. 베이스라인 드리프트는 신호의 기준축(zero-line)이 저주파 노이즈로 인해 위아래로 진동하는 현상으로, 신호 분석(심박수 계산, 이상 진단)을 방해합니다. 스파이크 아티팩트는 하드웨어 노이즈(전극 이동, glitch)로 인한 급격한 amp 탈출(갑작스러운 진폭 변화)로, 보정 신호가 왜곡되어 정보 손실(예: PQRST 피크 왜곡)이 발생합니다. 특히 866-875s 구간처럼 급격한 변화에서 보정 실패가 두드러지며, 전체 신호 품질 저하를 초래합니다.

## 원인 (근본)
- **베이스라인 드리프트 원인**: 저주파 노이즈(호흡, 전극 이동, 센서 드리프트)로 인해 신호의 DC offset이나 느린 변동이 발생. adaptive fitting 알고리즘(ASLS 기반 λ 계산)이 노이즈를 신호 변화로 오인하여 베이스라인을 과도하게 따라감 (노이즈 ↑ → λ ↓ → over-following).
- **스파이크 아티팩트 원인**: 하드웨어 impulse noise(고변동성 spike)로, 미분(gradient) 값이 크고 variance가 높아 fitting 취약. 기존 Masks(HV/Sag/Step)가 threshold 미흡하거나 연속 구간 처리를 못해, 보정 신호 amp 깨짐 발생. 본질적으로 알고리즘의 outlier handling 부족으로, spike가 전체 baseline 추정에 영향을 줌.

## 목표
- 베이스라인을 고정된 0선(zero-line)으로 안정화: 보정 신호(corrected)가 amp 0 주위로 대칭적으로 진동하도록 함 (드리프트/진동 제거).
- 스파이크 아티팩트 제거: 급격한 amp 탈출 구간(866-875s 등)에서 신호 왜곡 방지, 정보 손실 최소화.
- PQRST 피크 보존: 노이즈 제거 과정에서 ECG 핵심 특징(P-wave, QRS complex, T-wave) 왜곡 없음. 최종 신호 품질 향상으로 정확한 분석(피크 검출, HRV) 가능.
- 결론적으로 ECG Wave data를 정규화하여 데이터품질을 높이는 것이 목표
