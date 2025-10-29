# vis.py
# ECG 시각화 스크립트: vis/ 디렉토리에 플롯 저장
# 실행: python vis.py
# 요구사항: calibration.py의 함수(import 필요), JSON 파일 존재

import json
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
from pathlib import Path
import os

# calibration.py 함수 가져오기 (가능하면)
try:
    from calibration import baseline_hybrid_plus_adaptive as _bl_fn  # version1 (patched)
    from calibration import baseline_hybrid_plus_adaptive_original as _bl_v0  # version0 (original)
    from calibration import compute_drift_metric as _compute_drift_metric
    from calibration import extract_ecg as _extract_ecg_from_calibration
    # New strict zero-baseline variant from calibration.py
    try:
        from calibration import baseline_zero_drift as _bl_vz
    except Exception:
        _bl_vz = None
    try:
        from calibration_spicyyeol import baseline_hybrid_plus_adaptive as _bl_v2  # version2 (optimized)
    except Exception:
        _bl_v2 = None
except Exception:
    _bl_fn = None
    _bl_v0 = None
    _bl_v2 = None
    _bl_vz = None
    _compute_drift_metric = None
    _extract_ecg_from_calibration = None

# 로컬 fallback: calibration.py의 extract_ecg와 동일 동작
def extract_ecg(obj):
    if _extract_ecg_from_calibration is not None:
        try:
            return _extract_ecg_from_calibration(obj)
        except Exception:
            pass
    # fallback: 중첩 구조에서 'ECG' 리스트를 찾아 반환
    if isinstance(obj, dict):
        if 'ECG' in obj and isinstance(obj['ECG'], list):
            return np.array(obj['ECG'], dtype=float)
        for v in obj.values():
            hit = extract_ecg(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = extract_ecg(it)
            if hit is not None:
                return hit
    return None

FS = 250.0
FILE_PATH = Path("11646C1011258_test5_20250825T112545inPlainText.json")  # 사용자 지정 가능

# Drift metric fallback (calibration.py 미존재 시)
def compute_drift_metric(y_corr, b_final, fs):
    if _compute_drift_metric is not None:
        try:
            return _compute_drift_metric(y_corr, b_final, fs)
        except Exception:
            pass
    y_corr = np.asarray(y_corr, float)
    b_final = np.asarray(b_final, float)
    N = b_final.size
    if N == 0 or fs <= 0:
        return {
            'std_baseline': float('nan'),
            'mad_from_zero': float('nan'),
            'var_grad_baseline': float('nan'),
            'lf_power_frac_baseline': float('nan')
        }
    std_baseline = float(np.std(b_final))
    mad_from_zero = float(np.median(np.abs(y_corr)))
    var_grad_baseline = float(np.var(np.gradient(b_final)))
    try:
        from numpy.fft import rfft, rfftfreq
        B = rfft(b_final - float(np.nanmean(b_final)))
        f = rfftfreq(N, d=1.0/float(fs))
        P = (B.real*B.real + B.imag*B.imag)
        lf = f <= 0.5
        lf_power_frac = float(P[lf].sum() / (P.sum() + 1e-12))
    except Exception:
        lf_power_frac = float('nan')
    return {
        'std_baseline': std_baseline,
        'mad_from_zero': mad_from_zero,
        'var_grad_baseline': var_grad_baseline,
        'lf_power_frac_baseline': lf_power_frac,
    }

# version0 fallback: 원본 λ(노이즈↑→λ↓) 로직이 없을 경우, v1로 대체
def baseline_version0(y, fs, **kwargs):
    if _bl_v0 is not None:
        return _bl_v0(y, fs, **kwargs)
    if _bl_fn is not None:
        # Fallback: 비교 불가 시 v1 사용
        return _bl_fn(y, fs, **kwargs)
    # 최종 폴백
    y = np.asarray(y, float)
    return y.copy(), np.zeros_like(y)

def baseline_version1(y, fs, **kwargs):
    if _bl_fn is not None:
        return _bl_fn(y, fs, **kwargs)
    # 폴백
    y = np.asarray(y, float)
    return y.copy(), np.zeros_like(y)

def baseline_version2(y, fs, **kwargs):
    if _bl_v2 is not None:
        return _bl_v2(y, fs, **kwargs)
    # 폴백: v1 사용
    return baseline_version1(y, fs, **kwargs)

def baseline_versionZ(y, fs, cutoff=0.5, order=4, **kwargs):
    """Strict zero-baseline correction (Z): corrected ~ 0, baseline = drift"""
    if _bl_vz is not None:
        return _bl_vz(y, fs, cutoff=cutoff, order=order)
    # 폴백: 간단히 mean 제거만
    y = np.asarray(y, float)
    yc = y - float(np.nanmean(y))
    return yc, (y - yc)

# ---- argparse ----
parser = argparse.ArgumentParser(description='ECG baseline drift visualization with version comparison')
parser.add_argument('--version', choices=['0', '1', '2', 'z', 'both', 'all'], default='both', help='baseline version: 0(original), 1(patched), 2(optimized), z(strict zero-baseline), both(0&1), all(0&1&2&z)')
parser.add_argument('--file', type=str, default=str(FILE_PATH), help='input JSON file path')
parser.add_argument('--fs', type=float, default=FS, help='sampling rate (Hz)')
parser.add_argument('--fft-compare', action='store_true', help='also save band-wise Raw vs Corrected FFT compare plots')
parser.add_argument('--fft-log', action='store_true', help='use dB scale for FFT amplitude')
args = parser.parse_args()

# 입력 경로/FS 갱신
FILE_PATH = Path(args.file)
FS = float(args.fs)

# 데이터 로드
with FILE_PATH.open('r', encoding='utf-8') as f:
    data = json.load(f)
ecg_raw = extract_ecg(data)
assert ecg_raw is not None and ecg_raw.size > 0

# 중심화
y_raw = ecg_raw.astype(float)
y_raw = y_raw - float(np.nanmean(y_raw))

t = np.arange(len(y_raw)) / FS

# 출력 디렉토리 결정 (버전별)
if args.version == '0':
    OUT_DIR = os.path.join('vis', 'version0')
elif args.version == '1':
    OUT_DIR = os.path.join('vis', 'version1')
elif args.version == '2':
    OUT_DIR = os.path.join('vis', 'version2')
elif args.version == 'z':
    OUT_DIR = os.path.join('vis', 'versionZ')
else:  # both/all → 기본값은 v1 폴더(아래에서 버전별로 렌더링)
    OUT_DIR = os.path.join('vis', 'version1')
os.makedirs(OUT_DIR, exist_ok=True)

# 파일 베이스 이름: FILE_PATH stem의 앞부분
base_name = FILE_PATH.stem.split('_')[0]  # 예: 11646C1011258

# 공통 유틸
def _ensure_1d(a):
    return np.asarray(a, float).reshape(-1)

def _rolling_std_same(x, w):
    x = _ensure_1d(x)
    w = int(max(3, round(w)))
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=float)
    s1 = np.convolve(x, k, mode='same')
    s2 = np.convolve(x * x, k, mode='same')
    mean = s1 / w
    var = s2 / w - mean * mean
    var[var < 0] = 0.0
    return np.sqrt(var)

# 1) Time-domain plot of raw, estimated baseline, and corrected signal
def plot_time_triple(t, y_raw, b_final, y_corr, fs, save_path):
    t = _ensure_1d(t); y_raw = _ensure_1d(y_raw); b_final = _ensure_1d(b_final); y_corr = _ensure_1d(y_corr)
    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, y_raw, label='Raw', color='tab:gray', lw=1)
    ax1.axhline(0, color='k', lw=1, alpha=0.3)
    ax1.set_title('Raw Signal (zero-line shown)')
    ax1.set_ylabel('Amplitude'); ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, b_final, label='Estimated Baseline', color='tab:cyan', lw=1)
    ax2.axhline(0, color='k', lw=1, alpha=0.3)
    ax2.set_title('Estimated Baseline (drift inspection)')
    ax2.set_ylabel('Amplitude'); ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, y_corr, label='Corrected', color='tab:orange', lw=1.2)
    ax3.axhline(0, color='k', lw=1, alpha=0.3)
    ax3.set_title('Corrected Signal (should oscillate around zero)')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Amplitude'); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# 2) FFT spectrum of raw and corrected signals (한 이미지에 비교)
def plot_fft_raw_vs_corrected(y_raw, y_corr, fs, save_path):
    y_raw = _ensure_1d(y_raw); y_corr = _ensure_1d(y_corr)
    N = len(y_raw)
    fr = rfftfreq(N, 1.0 / fs)
    Yr = np.abs(rfft(y_raw)) * (2.0 / max(1, N))
    Yc = np.abs(rfft(y_corr)) * (2.0 / max(1, N))

    plt.figure(figsize=(12, 6))
    plt.plot(fr, Yr, label='Raw', color='tab:gray', lw=1)
    plt.plot(fr, Yc, label='Corrected', color='tab:orange', lw=1.2)
    # 저주파(0~0.5Hz) 밴드 강조
    plt.axvspan(0.0, 0.5, color='red', alpha=0.08, label='Low-freq band (≤0.5 Hz)')
    plt.title('FFT Spectrum — Raw vs Corrected')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, min(40, fs/2.0))  # ECG 관심대역 강조
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# 다중 대역 FFT 시각화 (Raw/Corrected 각각 + 선택적 비교)
def plot_fft_bands(y_raw, y_corr, fs, base_name, out_dir, compare=False, log_amp=False):
    """
    밴드별로 Raw(상단) / Corrected(하단)을 한 장에 묶어 총 4개 이미지를 저장.
    기존의 raw/ corrected 개별 저장(8개) 대신 통합 저장(4개)으로 변경.
    """
    y_raw = _ensure_1d(y_raw); y_corr = _ensure_1d(y_corr)
    N = len(y_raw)
    fr = rfftfreq(N, 1.0 / fs)
    Yr = np.abs(rfft(y_raw)) * (2.0 / max(1, N))
    Yc = np.abs(rfft(y_corr)) * (2.0 / max(1, N))

    bands = [
        (0.0, 0.5, 'noise'),
        (0.5, 5.0, 'low'),
        (5.0, 15.0, 'mid'),
        (15.0, 40.0, 'high'),
    ]

    def _maybe_log(a):
        if log_amp:
            return 20.0 * np.log10(np.maximum(a, 1e-12))
        return a

    for lo, hi, name in bands:
        m = (fr >= lo) & (fr <= hi)
        xf = fr[m]
        if xf.size == 0:
            continue
        amp_r = _maybe_log(Yr[m])
        amp_c = _maybe_log(Yc[m])

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # 상단: Raw
        ax_top.plot(xf, amp_r, color='tab:gray', lw=1)
        ax_top.set_title(f'FFT Raw — {name} band [{lo}-{hi} Hz]')
        ax_top.set_ylabel('Magnitude (dB)' if log_amp else 'Amplitude')
        ax_top.set_xlim(lo, hi)
        ax_top.grid(True, alpha=0.3)

        # 하단: Corrected
        ax_bot.plot(xf, amp_c, color='tab:orange', lw=1.2)
        ax_bot.set_title(f'FFT Corrected — {name} band [{lo}-{hi} Hz]')
        ax_bot.set_xlabel('Frequency (Hz)')
        ax_bot.set_ylabel('Magnitude (dB)' if log_amp else 'Amplitude')
        ax_bot.set_xlim(lo, hi)
        ax_bot.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{out_dir}/fft_{name}_{base_name}.png', dpi=140)
        plt.close(fig)

# 3) PSD of corrected signal (Welch)
def plot_psd_corrected(y_corr, fs, save_path):
    y_corr = _ensure_1d(y_corr)
    nperseg = max(256, min(4096, len(y_corr)//2 if len(y_corr) > 0 else 256))
    f, Pxx = welch(y_corr, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend='constant', scaling='density')
    plt.figure(figsize=(12, 5))
    plt.semilogy(f, Pxx, color='tab:orange', lw=1.2)
    # 저주파(0~0.5Hz) 밴드 강조
    plt.axvspan(0.0, 0.5, color='red', alpha=0.08, label='Low-freq band (≤0.5 Hz)')
    plt.title('PSD (Welch) — Corrected Signal')
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Power Spectral Density')
    plt.xlim(0, min(40, fs/2.0))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# 4) Histogram of residuals (y_raw - b_final)
def plot_hist_residuals(y_raw, b_final, save_path):
    y_raw = _ensure_1d(y_raw); b_final = _ensure_1d(b_final)
    resid = y_raw - b_final
    mu = float(np.mean(resid))
    med = float(np.median(resid))
    plt.figure(figsize=(8, 5))
    plt.hist(resid, bins=100, color='tab:blue', alpha=0.7, edgecolor='none')
    plt.axvline(0, color='k', lw=1, alpha=0.6, label='Zero')
    plt.axvline(mu, color='tab:red', ls='--', lw=1, alpha=0.8, label=f'Mean={mu:.3g}')
    plt.axvline(med, color='tab:green', ls='-.', lw=1, alpha=0.8, label=f'Median={med:.3g}')
    plt.title('Histogram of Residuals (y_raw - baseline)')
    plt.xlabel('Residual amplitude'); plt.ylabel('Count')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# 5) Rolling standard deviation of the baseline
def plot_rolling_std_baseline(t, b_final, fs, window_s, save_path):
    t = _ensure_1d(t); b_final = _ensure_1d(b_final)
    w = int(max(3, round(window_s * fs)))
    rs = _rolling_std_same(b_final, w)
    plt.figure(figsize=(12, 4.5))
    plt.plot(t, rs, color='tab:purple', lw=1.2)
    plt.title(f'Rolling Std of Baseline (window={window_s:.2f}s)')
    plt.xlabel('Time (s)'); plt.ylabel('Std (a.u.)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# === 실행 ===
parms = dict(
    per_win_s=2.8, per_q=15,
    asls_lam=8e7, asls_p=0.01, asls_decim=12,
    qrs_aware=True, verylow_fc=0.03, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    hard_cut=True, break_pad_s=0.30
)

# 버전 실행/메트릭 계산
run_both = (args.version == 'both')
run_all = (args.version == 'all')
y_corr_v0 = b_v0 = y_corr_v1 = b_v1 = y_corr_v2 = b_v2 = y_corr_vz = b_vz = None
m0 = m1 = m2 = mz = None

if run_both or run_all or args.version == '0':
    y_corr_v0, b_v0 = baseline_version0(y_raw, FS, **parms)
    m0 = compute_drift_metric(y_corr_v0, b_v0, FS)

if run_both or run_all or args.version == '1':
    y_corr_v1, b_v1 = baseline_version1(y_raw, FS, **parms)
    m1 = compute_drift_metric(y_corr_v1, b_v1, FS)

if run_all or args.version == '2':
    y_corr_v2, b_v2 = baseline_version2(y_raw, FS, **parms)
    m2 = compute_drift_metric(y_corr_v2, b_v2, FS)

if run_all or args.version == 'z':
    # Z 버전은 cutoff/order를 명시적으로 사용(필요 시 파라미터화 가능)
    y_corr_vz, b_vz = baseline_versionZ(y_raw, FS, cutoff=0.5, order=4)
    mz = compute_drift_metric(y_corr_vz, b_vz, FS)

# 비교 출력
def _fmt(v):
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)

if (run_both or run_all) and (m0 is not None) and (m1 is not None):
    keys = ['std_baseline', 'mad_from_zero', 'var_grad_baseline', 'lf_power_frac_baseline']
    print("\nBaseline metrics comparison (lower is better):")
    header = f"{'metric':28s}  {'v0':>14s}  {'v1':>14s}  {'improve%':>10s}"
    print(header)
    print('-'*len(header))
    for k in keys:
        v0 = float(m0.get(k, np.nan))
        v1 = float(m1.get(k, np.nan))
        denom = (abs(v0) + 1e-12)
        imp = 100.0 * (v0 - v1) / denom
        print(f"{k:28s}  {v0:14.6g}  {v1:14.6g}  {imp:10.2f}")

# 추가 비교 콘솔 출력: v0-v2, v1-v2
if run_all and (m0 is not None) and (m2 is not None):
    keys = ['std_baseline', 'mad_from_zero', 'var_grad_baseline', 'lf_power_frac_baseline']
    print("\nBaseline metrics comparison (v0 vs v2, lower is better):")
    header = f"{'metric':28s}  {'v0':>14s}  {'v2':>14s}  {'improve%':>10s}"
    print(header)
    print('-'*len(header))
    for k in keys:
        v0 = float(m0.get(k, np.nan))
        v2_ = float(m2.get(k, np.nan))
        denom = (abs(v0) + 1e-12)
        imp = 100.0 * (v0 - v2_) / denom
        print(f"{k:28s}  {v0:14.6g}  {v2_:14.6g}  {imp:10.2f}")

if run_all and (m1 is not None) and (m2 is not None):
    keys = ['std_baseline', 'mad_from_zero', 'var_grad_baseline', 'lf_power_frac_baseline']
    print("\nBaseline metrics comparison (v1 vs v2, lower is better):")
    header = f"{'metric':28s}  {'v1':>14s}  {'v2':>14s}  {'improve%':>10s}"
    print(header)
    print('-'*len(header))
    for k in keys:
        v1 = float(m1.get(k, np.nan))
        v2_ = float(m2.get(k, np.nan))
        denom = (abs(v1) + 1e-12)
        imp = 100.0 * (v1 - v2_) / denom
        print(f"{k:28s}  {v1:14.6g}  {v2_:14.6g}  {imp:10.2f}")

# 어떤 결과로 시각화할지 선택: both/all은 버전별 폴더에 각각 저장
if run_both or run_all:
    y_corr, b_final = y_corr_v1, b_v1
elif args.version == '0':
    y_corr, b_final = y_corr_v0, b_v0
elif args.version == '2':
    y_corr, b_final = y_corr_v2, b_v2
elif args.version == 'z':
    y_corr, b_final = y_corr_vz, b_vz
else:
    y_corr, b_final = y_corr_v1, b_v1

# ---- 구간별 메트릭 CSV 저장 ----
def _fixed_segments_by_seconds(n, fs, start_s=1000.0, end_s=3500.0, step_s=500.0):
    i0 = max(0, int(round(start_s * fs)))
    i_end = min(n, int(round(end_s * fs)))
    step = max(1, int(round(step_s * fs)))
    i = i0
    while i < i_end:
        j = min(i_end, i + step)
        yield i, j
        i = j

def save_segment_metrics_csv(base_name, fs, versions: dict,
                             start_s=1000.0, end_s=3500.0, step_s=500.0,
                             out_dir: str = 'vis'):
    """
    versions: { 'v0': (y_corr, b_final), 'v1': (y_corr, b_final), ... }
    vis/metrics_segments_{base_name}.csv (버전별 행) 덮어쓰기
    vis/metrics_segments_compare_v0_v1_{base_name}.csv (둘 다 있을 때 비교) 덮어쓰기
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(next(iter(versions.values()))[0]) if versions else 0
    keys = ['std_baseline', 'mad_from_zero', 'var_grad_baseline', 'lf_power_frac_baseline']

    # 1) 버전별 세그먼트 메트릭(롱 포맷)
    out_path = f'{out_dir}/metrics_segments_{base_name}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['version', 'seg_idx', 't_start', 't_end'] + keys)
        for ver_name, (yc, bf) in versions.items():
            for k, (i0, i1) in enumerate(_fixed_segments_by_seconds(N, fs, start_s, end_s, step_s)):
                t0 = i0 / fs; t1 = i1 / fs
                met = compute_drift_metric(yc[i0:i1], bf[i0:i1], fs)
                row = [ver_name, k, f"{t0:.6f}", f"{t1:.6f}"] + [met.get(kx, np.nan) for kx in keys]
                w.writerow(row)

    # 2) 버전쌍 비교 CSV들 생성(존재하는 모든 조합)
    ver_keys = list(versions.keys())
    pairs = []
    for i in range(len(ver_keys)):
        for j in range(i+1, len(ver_keys)):
            pairs.append((ver_keys[i], ver_keys[j]))

    for va, vb in pairs:
        out_cmp = f'{out_dir}/metrics_segments_compare_{va}_{vb}_{base_name}.csv'
        with open(out_cmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            yca, bfa = versions[va]; ycb, bfb = versions[vb]

            segs = list(_fixed_segments_by_seconds(N, fs, start_s, end_s, step_s))
            seg_idx = list(range(len(segs)))
            t_start = []
            t_end = []
            vals_va = {k: [] for k in keys}
            vals_vb = {k: [] for k in keys}
            vals_imp = {k: [] for k in keys}
            for (i0, i1) in segs:
                t0 = i0 / fs; t1 = i1 / fs
                t_start.append(t0)
                t_end.append(t1)
                mas = compute_drift_metric(yca[i0:i1], bfa[i0:i1], fs)
                mbs = compute_drift_metric(ycb[i0:i1], bfb[i0:i1], fs)
                for kx in keys:
                    va_ = float(mas.get(kx, np.nan))
                    vb_ = float(mbs.get(kx, np.nan))
                    denom = (abs(va_) + 1e-12)
                    imp = 100.0 * (va_ - vb_) / denom
                    vals_va[kx].append(va_)
                    vals_vb[kx].append(vb_)
                    vals_imp[kx].append(imp)

            w.writerow(['seg_idx'] + seg_idx)
            w.writerow(['t_start'] + [f"{v:.6f}" for v in t_start])
            w.writerow(['t_end'] + [f"{v:.6f}" for v in t_end])
            for kx in keys:
                w.writerow([f'{kx}_{va}'] + vals_va[kx])
                w.writerow([f'{kx}_{vb}'] + vals_vb[kx])
            w.writerow([''] + ['--------------'] * len(seg_idx))
            for kx in keys:
                w.writerow([f'{kx}_improve_pct'] + vals_imp[kx])

# 버전 dict 구성 및 저장 호출
versions_dict = {}
if run_both or run_all or args.version == '0':
    if y_corr_v0 is not None and b_v0 is not None:
        versions_dict['v0'] = (y_corr_v0, b_v0)
if run_both or run_all or args.version == '1':
    if y_corr_v1 is not None and b_v1 is not None:
        versions_dict['v1'] = (y_corr_v1, b_v1)
if run_all or args.version == '2':
    if y_corr_v2 is not None and b_v2 is not None:
        versions_dict['v2'] = (y_corr_v2, b_v2)
if run_all or args.version == 'z':
    if y_corr_vz is not None and b_vz is not None:
        versions_dict['vZ'] = (y_corr_vz, b_vz)

# 메트릭 CSV는 버전 공통 비교를 위해 vis/ 루트에 저장
save_segment_metrics_csv(base_name, FS, versions_dict,
                         start_s=1000.0, end_s=3500.0, step_s=500.0,
                         out_dir='vis')

def _render_all_plots(y_raw, y_corr, b_final, t, fs, base_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 1) Time-domain triple (raw, baseline, corrected)
    plot_time_triple(
        t, y_raw, b_final, y_corr, fs,
        save_path=f'{out_dir}/time_raw-baseline-corrected_{base_name}.png'
    )
    # 2) FFT — 대역별 Raw/Corrected(상/하단) 통합 플롯
    plot_fft_bands(
        y_raw, y_corr, fs, base_name, out_dir,
        compare=args.fft_compare, log_amp=args.fft_log
    )
    # 3) PSD — Corrected (Welch)
    plot_psd_corrected(
        y_corr, fs,
        save_path=f'{out_dir}/psd_corrected_{base_name}.png'
    )
    # 4) Histogram of residuals
    plot_hist_residuals(
        y_raw, b_final,
        save_path=f'{out_dir}/hist_residuals_{base_name}.png'
    )
    # 5) Rolling std of baseline
    plot_rolling_std_baseline(
        t, b_final, fs, window_s=5.0,
        save_path=f'{out_dir}/rollingstd_baseline_{base_name}.png'
    )

# 렌더링: both/all면 버전별 모두 생성, 단일이면 해당 버전만 생성
if run_both or run_all:
    out_v0 = os.path.join('vis', 'version0')
    out_v1 = os.path.join('vis', 'version1')
    out_v2 = os.path.join('vis', 'version2')
    out_vZ = os.path.join('vis', 'versionZ')
    if y_corr_v0 is not None and b_v0 is not None:
        _render_all_plots(y_raw, y_corr_v0, b_v0, t, FS, base_name, out_v0)
    if y_corr_v1 is not None and b_v1 is not None:
        _render_all_plots(y_raw, y_corr_v1, b_v1, t, FS, base_name, out_v1)
    if run_all and (y_corr_v2 is not None and b_v2 is not None):
        _render_all_plots(y_raw, y_corr_v2, b_v2, t, FS, base_name, out_v2)
    if run_all and (y_corr_vz is not None and b_vz is not None):
        _render_all_plots(y_raw, y_corr_vz, b_vz, t, FS, base_name, out_vZ)
    if run_all:
        print(f"\nVisualizations saved in {out_v0}/, {out_v1}/, {out_v2}/ and {out_vZ}/")
    else:
        print(f"\nVisualizations saved in {out_v0}/ and {out_v1}/")
else:
    _render_all_plots(y_raw, y_corr, b_final, t, FS, base_name, OUT_DIR)
    print(f"\nVisualizations saved in {OUT_DIR}/")
print(
    "\nWhy these 5 visualizations (zero-line checks):\n"
    "1) Time-domain (raw/baseline/corrected): 눈으로 드리프트와 보정 결과의 0선 대칭성 확인\n"
    "2) FFT Raw vs Corrected: 0~0.5Hz 저주파 에너지 감소로 드리프트 억제 여부 정량 확인\n"
    "3) PSD (Welch) Corrected: 스무딩된 주파수 뷰로 잔존 저주파 파워 검출(미세 드리프트 파악)\n"
    "4) Residuals Histogram: (y_raw - baseline)의 분포가 0 중심 대칭인지로 오프셋/바이어스 확인\n"
    "5) Baseline Rolling Std: 시간대별 기준선 불안정(출렁임) 구간 식별 및 디버깅"
)
if run_all and (mz is not None):
    keys = ['std_baseline', 'mad_from_zero', 'var_grad_baseline', 'lf_power_frac_baseline']
    print("\nBaseline metrics comparison (v2 vs vZ, lower is better):")
    header = f"{'metric':28s}  {'v2':>14s}  {'vZ':>14s}  {'improve%':>10s}"
    print(header)
    print('-'*len(header))
    for k in keys:
        v2_ = float((m2 or {}).get(k, np.nan))
        vz_ = float(mz.get(k, np.nan))
        denom = (abs(v2_) + 1e-12)
        imp = 100.0 * (v2_ - vz_) / denom
        print(f"{k:28s}  {v2_:14.6g}  {vz_:14.6g}  {imp:10.2f}")
