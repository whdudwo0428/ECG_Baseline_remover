import json
import numpy as np
from pathlib import Path

def _extract_ecg(obj):
    """중첩 JSON 어디에 있든 'ECG' 리스트를 찾아 ndarray로 반환."""
    if isinstance(obj, dict):
        if 'ECG' in obj and isinstance(obj['ECG'], list):
            return np.array(obj['ECG'], dtype=float)
        for v in obj.values():
            hit = _extract_ecg(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = _extract_ecg(it)
            if hit is not None:
                return hit
    return None

def _resample_to_250(x, fs_raw, fs_tgt=250.0):
    """scipy가 있으면 resample_poly/decimate, 없으면 근사 블록평균 폴백."""
    try:
        from fractions import Fraction
        from scipy.signal import resample_poly
        frac = Fraction(fs_tgt, fs_raw).limit_denominator(100)
        up, down = frac.numerator, frac.denominator
        y = resample_poly(x, up, down)
        return y
    except Exception:
        # 정수 비일 때는 블록 평균/샘플링 폴백
        q = int(round(fs_raw / fs_tgt))
        if q > 1 and abs(fs_raw / q - fs_tgt) / fs_tgt < 0.02:
            n = (len(x)//q) * q
            return x[:n].reshape(-1, q).mean(axis=1)
        # 마지막 폴백: 최근접 샘플링
        idx = np.round(np.linspace(0, len(x)-1, int(len(x) * fs_tgt / fs_raw))).astype(int)
        return x[idx]

def _rolling_std(y, w):
    """컨볼루션 기반 O(N) 롤링 표준편차 (same 길이)."""
    y = y.astype(float)
    w = int(w)
    k = np.ones(w, dtype=float)
    s1 = np.convolve(y, k, mode='same')
    s2 = np.convolve(y*y, k, mode='same')
    mean = s1 / w
    var = s2 / w - mean*mean
    var[var < 0] = 0.0
    return np.sqrt(var)

def process_ecg_json_to_interpolated(
    json_path,
    fs_raw,
    fs_out=250.0,
    win_samples=2000,
    k_sigma=20.0,
    pad_samples=125,
    sat_lo=0.0,
    sat_hi=4095.0
):
    """
    JSON의 ECG를 읽어 250Hz로 리샘플 후, 고변동 구간을 보간해 제거한 신호를 반환.
    Parameters
    ----------
    json_path : str | Path
        'ECG' 배열을 포함한 JSON 파일 경로
    fs_raw : float
        원본 샘플링 주파수(Hz)
    fs_out : float, default 250.0
        목표 샘플링 주파수(Hz) – 요구사항: 250
    win_samples : int, default 2000
        롤링 STD 윈도 크기(샘플) – 250Hz 기준 약 8초
    k_sigma : float, default 20.0
        임계 = median(rollSTD) + k_sigma * MAD(rollSTD)
    pad_samples : int, default 125
        검출 구간 양옆 확장(샘플) – 250Hz 기준 0.5초
    sat_lo, sat_hi : float
        포화값 하/상한. (장비 ADC 범위에 맞춰 조정 가능)
    Returns
    -------
    t : np.ndarray
        시간축(초), 길이 = 출력 신호 길이
    y_interp : np.ndarray
        고변동 구간을 NaN→선형보간으로 메운 동일 길이 신호(250Hz)
    """
    json_path = Path(json_path)
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    raw = _extract_ecg(data)
    if raw is None or raw.size == 0:
        raise ValueError("JSON에서 'ECG' 배열을 찾지 못했습니다.")

    # 1) 250 Hz 리샘플
    y = _resample_to_250(raw.astype(float), fs_raw, fs_tgt=fs_out)
    fs = float(fs_out)
    N = y.size
    t = np.arange(N) / fs

    # 2) 포화/비정상 값 NaN 처리 후 통계 계산용 채움
    x = y.copy()
    nan_mask = (x <= sat_lo) | (x >= sat_hi) | ~np.isfinite(x)
    x[nan_mask] = np.nan
    med = np.nanmedian(x)
    x_filled = x.copy()
    x_filled[np.isnan(x_filled)] = med

    # 3) 롤링 STD -> 임계값(robust) -> 고변동 마스크
    win = max(2, int(win_samples))
    rs = _rolling_std(x_filled, win)
    rs_med = np.median(rs)
    rs_mad = np.median(np.abs(rs - rs_med)) + 1e-12
    thr = rs_med + 1.4826 * rs_mad * float(k_sigma)
    high_var = rs > thr

    # 4) 경계 확장
    pad = int(max(0, pad_samples))
    if pad > 0:
        padk = np.ones(pad*2 + 1, dtype=int)
        high_var = (np.convolve(high_var.astype(int), padk, mode='same') > 0)

    # 5) 동일 길이 보간 (NaN → 선형보간, 가장자리 최근접)
    y_interp = y.astype(float).copy()
    y_interp[high_var] = np.nan
    good = ~np.isnan(y_interp)
    if good.sum() >= 2:
        first = int(np.argmax(good))
        last = int(len(y_interp) - 1 - np.argmax(good[::-1]))
        if first > 0:
            y_interp[:first] = y_interp[first]
        if last < len(y_interp) - 1:
            y_interp[last+1:] = y_interp[last]
        good = ~np.isnan(y_interp)
        y_interp[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), y_interp[good])
    else:
        y_interp[:] = med

    return t, y_interp

if __name__=="__main__":
    '''
    t : 출력신호의 시간축
    y_interp : 결과 신호
    '''
    t, clean = process_ecg_json_to_interpolated(
        "11636C1011258_test1_20250825T111732inPlainText.json",
        fs_raw=1000.0,
        fs_out=250.0,  # 고정 요구사항
        win_samples=2000,  # 요구사항
        k_sigma=20.0,  # 요구사항
        pad_samples=125  # 0.5초(선택)
    )
