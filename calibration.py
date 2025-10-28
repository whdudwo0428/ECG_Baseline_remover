# -*- coding: utf-8 -*-
# ECG Viewer — 1000→250 Hz | Hybrid BL++ (adaptive λ, variance-aware, hard-cut) + Residual Refit
# (AGC & Glitch 제거 버전)
# Masks(Sag/Step/Corner/Burst/Wave/HV)는 PROCESSED 신호(y_corr_eq=y_corr) 기준. 보간 없음.

import json, sys
from pathlib import Path
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# =========================
# Config
# =========================
FILE_PATH = Path('11646C1011258_test5_20250825T112545inPlainText.json')
FS_RAW = 250.0
FS = 250.0
DECIM = int(round(FS_RAW / FS)) if FS > 0 else 1
if DECIM < 1: DECIM = 1

# =========================
# IO & Utils
# =========================
def extract_ecg(obj):
    if isinstance(obj, dict):
        if 'ECG' in obj and isinstance(obj['ECG'], list):
            return np.array(obj['ECG'], dtype=float)
        for v in obj.values():
            hit = extract_ecg(v)
            if hit is not None: return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = extract_ecg(it)
            if hit is not None: return hit
    return None

def decimate_fir_zero_phase(x, q=4):
    from scipy.signal import decimate
    return decimate(x, q, ftype='fir', zero_phase=True)

def decimate_if_needed(x, decim: int):
    if decim <= 1: return x
    try:
        return decimate_fir_zero_phase(x, decim)
    except Exception:
        n = (len(x)//decim)*decim
        return x[:n].reshape(-1, decim).mean(axis=1)

def _onepole(sig, fc, fs):
    if fc <= 0: return sig
    sig = np.asarray(sig, float)
    beta = (2*np.pi*fc) / (2*np.pi*fc + fs)
    y = np.empty_like(sig, dtype=float)
    y[0] = float(sig[0])
    for i in range(1, y.size):
        y[i] = y[i-1] + beta*(sig[i] - y[i-1])
    return y

def remove_baseline_drift(y, fs, cutoff=0.5, order=4):
    """
    하이패스(≤cutoff Hz 제거)로 드리프트를 제거하여 0선 기준으로 고정.
    zero-phase 적용, 출력은 평균 0으로 재중심화.
    """
    y = np.asarray(y, float)
    if y.size == 0 or fs <= 0:
        return y
    try:
        from scipy.signal import butter, sosfiltfilt
        wn = float(cutoff) / max(1e-12, fs * 0.5)
        wn = min(max(wn, 1e-6), 0.999999)
        sos = butter(int(order), wn, btype='highpass', output='sos')
        z = sosfiltfilt(sos, y)
    except Exception:
        # 폴백: 느린 1차 로우패스를 빼는 방식
        z = y - _onepole(y, fc=cutoff, fs=fs)
    z = z - float(np.nanmean(z))
    return z

# =========================
# Baseline core deps
# =========================
def baseline_asls_masked(y, lam=1e6, p=0.008, niter=10, mask=None,
                         cg_tol=1e-3, cg_maxiter=200, decim_for_baseline=1):
    from scipy.sparse.linalg import cg, LinearOperator
    y = np.asarray(y, float); N = y.size
    if N < 3: return np.zeros_like(y)
    if decim_for_baseline > 1:
        q = int(decim_for_baseline)
        n = (N // q) * q
        y_head = y[:n]; y_ds = y_head.reshape(-1, q).mean(axis=1)
        z_ds = baseline_asls_masked(y_ds, lam=lam, p=p, niter=niter, mask=None,
                                    cg_tol=cg_tol, cg_maxiter=cg_maxiter, decim_for_baseline=1)
        idx = np.repeat(np.arange(z_ds.size), q)
        z_coarse = z_ds[idx]
        if z_coarse.size < N:
            z = np.empty(N, float); z[:z_coarse.size] = z_coarse; z[z_coarse.size:] = z_coarse[-1]
        else: z = z_coarse[:N]
        return z
    g = np.ones(N) if mask is None else np.where(mask, 1.0, 1e-3)
    kernel = np.array([1., -4., 6., -4., 1.], float)
    def solve_once(w):
        wg = w * g; b = wg * y
        def mv(v):
            Av = wg * v
            Av += lam * np.convolve(v, kernel, mode='same')
            return Av
        A = LinearOperator((N, N), matvec=mv, dtype=float)
        z, info = cg(A, b, tol=cg_tol, maxiter=cg_maxiter)
        if info != 0:
            z, _ = cg(A, b, tol=max(cg_tol*5, 5e-3), maxiter=cg_maxiter*2)
        return z
    w = np.ones(N); z = np.zeros(N)
    for _ in range(niter):
        z = solve_once(w)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def make_qrs_mask(y, fs=250, r_pad_ms=180, t_pad_start_ms=80, t_pad_end_ms=300):
    import neurokit2 as nk
    info = nk.ecg_peaks(y, sampling_rate=fs)[1]
    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
    mask = np.ones_like(y, dtype=bool)
    if r_idx.size == 0: return mask
    def clamp(a): return np.clip(a, 0, len(y)-1)
    r_pad = int(round(r_pad_ms * 1e-3 * fs))
    t_s   = int(round(t_pad_start_ms * 1e-3 * fs))
    t_e   = int(round(t_pad_end_ms   * 1e-3 * fs))
    for r in r_idx:
        mask[clamp(r-r_pad):clamp(r+r_pad)+1] = False
        mask[clamp(r+t_s):clamp(r+t_e)+1]     = False
    return mask

# 변화점 탐지/마스크 팽창
def _find_breaks(y, fs, k=7.0, min_gap_s=0.30):
    dy = np.diff(y, prepend=y[0])
    med = np.median(dy); mad = np.median(np.abs(dy - med)) + 1e-12
    z = np.abs(dy - med) / (1.4826 * mad)
    idx = np.flatnonzero(z > float(k))
    if idx.size == 0: return []
    gap = int(round(min_gap_s * fs))
    breaks = [int(idx[0])]
    for i in idx[1:]:
        if i - breaks[-1] > gap:
            breaks.append(int(i))
    return breaks

def _dilate_mask(mask, fs, pad_s=0.45):
    pad = int(round(pad_s * fs))
    if pad <= 0: return mask
    k = np.ones(pad*2+1, dtype=int)
    return (np.convolve(mask.astype(int), k, mode='same') > 0)

# Hybrid BL++ (adaptive λ, variance-aware, hard-cut, local refit)
def baseline_hybrid_plus_adaptive(
    y, fs,
    per_win_s=2.8, per_q=15,
    asls_lam=1e8, asls_p=0.01, asls_decim=12,
    qrs_aware=True, verylow_fc=0.03, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    hard_cut=True, break_pad_s=0.30
):
    from scipy.ndimage import percentile_filter, median_filter

    x = np.asarray(y, float); N = x.size
    if N < 8:
        return np.zeros_like(x), np.zeros_like(x)

    # 0) 초기 퍼센타일 바닥선
    w = max(3, int(round(per_win_s * fs)));  w += (w % 2 == 0)
    dc = np.median(x[np.isfinite(x)])
    x0 = x - dc
    b0 = percentile_filter(x0, percentile=per_q, size=w, mode='nearest')

    # 1) QRS-aware + 변화점 보호
    try:
        base_mask = make_qrs_mask(x, fs=fs) if qrs_aware else np.ones_like(x, bool)
    except Exception:
        base_mask = np.ones_like(x, bool)
    brks = _find_breaks(x, fs, k=6.5, min_gap_s=0.25)
    prot = np.zeros_like(x, bool)
    if len(brks) > 0:
        prot[brks] = True
        prot = _dilate_mask(prot, fs, pad_s=max(0.35, break_pad_s))
        base_mask = base_mask & (~prot)

    # 2) 위치별 λ 설계 (gradient + volatility)
    grad = np.gradient(x)
    g_ref = np.percentile(np.abs(grad), 95) + 1e-6
    z_grad = np.clip(np.abs(grad) / g_ref, 0, 6.0)
    lam_grad = asls_lam / (1.0 + 8.0 * z_grad)

    vw = max(5, int(round(vol_win_s * fs)));  vw += (vw % 2 == 0)
    k = np.ones(vw, float)
    s1 = np.convolve(x, k, mode='same'); s2 = np.convolve(x*x, k, mode='same')
    m = s1 / vw; v = s2 / vw - m*m; v[v < 0] = 0.0
    rs = np.sqrt(v)
    rs_ref = np.percentile(rs, 90) + 1e-9
    z_vol = np.clip(rs / rs_ref, 0, 10.0)
    lam_vol = asls_lam / (1.0 + float(vol_gain) * z_vol)

    lam_local = np.minimum(lam_grad, lam_vol)
    if len(brks) > 0:
        tw = int(round(0.6 * fs))
        for b in brks:
            lo = max(0, b - tw); hi = min(N, b + tw + 1)
            lam_local[lo:hi] = np.minimum(lam_local[lo:hi], asls_lam * max(5e-4, lam_floor_ratio*0.5))
    lam_local = np.maximum(lam_local, asls_lam * max(5e-4, lam_floor_ratio))

    # 3) 세그먼트 피팅
    b1 = np.zeros_like(x)
    if len(brks) == 0 or not hard_cut:
        step = max(1, int(0.35 * fs))
        for i in range(0, N, step):
            j = min(N, i+step)
            lam_i = float(np.median(lam_local[i:j]))
            seg = x0[i:j] - b0[i:j]
            mask_i = None if not qrs_aware else base_mask[i:j]
            b1[i:j] = baseline_asls_masked(seg, lam=max(5e4, lam_i), p=asls_p,
                                           niter=10, mask=mask_i, decim_for_baseline=max(1, int(asls_decim)))
    else:
        cuts = [0] + [int(c) for c in brks] + [N]
        for k_i in range(len(cuts)-1):
            s0, e0 = cuts[k_i], cuts[k_i+1]
            if (e0 - s0) < int(0.5 * fs):
                if k_i < len(cuts)-2: cuts[k_i+1] = e0 = cuts[k_i+2]
                else: s0 = max(0, s0 - int(0.25 * fs))
            s, e = s0, e0
            lam_i = float(np.median(lam_local[s:e]))
            seg = x0[s:e] - b0[s:e]
            mask_i = None if not qrs_aware else base_mask[s:e]
            b1_seg = baseline_asls_masked(seg, lam=max(3e4, lam_i), p=asls_p,
                                          niter=10, mask=mask_i, decim_for_baseline=max(1, int(asls_decim)))
            b1[s:e] = b1_seg
        # 변화점 ±pad 초 로컬 리핏
        pad = int(round(break_pad_s * fs))
        if pad > 0:
            from scipy.ndimage import percentile_filter as _pf, median_filter as _mf
            for b in brks:
                lo = max(0, b - pad); hi = min(N, b + pad + 1)
                wloc = max(3, int(round(0.35 * fs))); wloc += (wloc % 2 == 0)
                resid = (x0[lo:hi] - b0[lo:hi] - b1[lo:hi])
                b_loc = _pf(resid, percentile=20, size=wloc, mode='nearest')
                b_loc = _mf(b_loc, size=max(3, int(round(0.12*fs))), mode='nearest')
                b1[lo:hi] += b_loc

    # 4) 매우저주파 안정화 + 잔차 클램프
    b = b0 + b1
    b_slow = _onepole(b, verylow_fc, fs)
    clamp_w = max(3, int(round(clamp_win_s * fs))); clamp_w += (clamp_w % 2 == 0)
    from scipy.ndimage import median_filter
    resid = x - b_slow
    off = median_filter(resid, size=clamp_w, mode='nearest')
    b_final = b_slow + off
<<<<<<< Updated upstream
    y_corr = x - b_final
    return y_corr, b_final
=======
    y_corr = x - b_final
    return y_corr, b_final

# 원래 λ 의존(노이즈↑→λ↓)을 유지한 비교용 함수
def baseline_hybrid_plus_adaptive_original(
    y, fs,
    per_win_s=2.8, per_q=15,
    asls_lam=1e8, asls_p=0.01, asls_decim=12,
    qrs_aware=True, verylow_fc=0.03, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    hard_cut=True, break_pad_s=0.30
):
    from scipy.ndimage import percentile_filter, median_filter

    x = np.asarray(y, float); N = x.size
    if N < 8:
        return np.zeros_like(x), np.zeros_like(x)

    # 0) 초기 퍼센타일 바닥선
    w = max(3, int(round(per_win_s * fs)));  w += (w % 2 == 0)
    dc = np.median(x[np.isfinite(x)])
    x0 = x - dc
    b0 = percentile_filter(x0, percentile=per_q, size=w, mode='nearest')

    # 1) QRS-aware + 변화점 보호
    try:
        base_mask = make_qrs_mask(x, fs=fs) if qrs_aware else np.ones_like(x, bool)
    except Exception:
        base_mask = np.ones_like(x, bool)
    brks = _find_breaks(x, fs, k=6.5, min_gap_s=0.25)
    prot = np.zeros_like(x, bool)
    if len(brks) > 0:
        prot[brks] = True
        prot = _dilate_mask(prot, fs, pad_s=max(0.35, break_pad_s))
        base_mask = base_mask & (~prot)

    # 2) 위치별 λ 설계 (gradient + volatility)
    grad = np.gradient(x)
    g_ref = np.percentile(np.abs(grad), 95) + 1e-6
    z_grad = np.clip(np.abs(grad) / g_ref, 0, 6.0)
    lam_grad = asls_lam / (1.0 + 8.0 * z_grad)

    vw = max(5, int(round(vol_win_s * fs)));  vw += (vw % 2 == 0)
    k = np.ones(vw, float)
    s1 = np.convolve(x, k, mode='same'); s2 = np.convolve(x*x, k, mode='same')
    m = s1 / vw; v = s2 / vw - m*m; v[v < 0] = 0.0
    rs = np.sqrt(v)
    rs_ref = np.percentile(rs, 90) + 1e-9
    z_vol = np.clip(rs / rs_ref, 0, 10.0)
    lam_vol = asls_lam / (1.0 + float(vol_gain) * z_vol)

    lam_local = np.minimum(lam_grad, lam_vol)
    if len(brks) > 0:
        tw = int(round(0.6 * fs))
        for b in brks:
            lo = max(0, b - tw); hi = min(N, b + tw + 1)
            lam_local[lo:hi] = np.minimum(lam_local[lo:hi], asls_lam * max(5e-4, lam_floor_ratio*0.5))
    lam_local = np.maximum(lam_local, asls_lam * max(5e-4, lam_floor_ratio))

    # 3) 세그먼트 피팅
    b1 = np.zeros_like(x)
    if len(brks) == 0 or not hard_cut:
        step = max(1, int(0.35 * fs))
        for i in range(0, N, step):
            j = min(N, i+step)
            lam_i = float(np.median(lam_local[i:j]))
            seg = x0[i:j] - b0[i:j]
            mask_i = None if not qrs_aware else base_mask[i:j]
            b1[i:j] = baseline_asls_masked(seg, lam=max(5e4, lam_i), p=asls_p,
                                           niter=10, mask=mask_i, decim_for_baseline=max(1, int(asls_decim)))
    else:
        cuts = [0] + [int(c) for c in brks] + [N]
        for k_i in range(len(cuts)-1):
            s0, e0 = cuts[k_i], cuts[k_i+1]
            if (e0 - s0) < int(0.5 * fs):
                if k_i < len(cuts)-2: cuts[k_i+1] = e0 = cuts[k_i+2]
                else: s0 = max(0, s0 - int(0.25 * fs))
            s, e = s0, e0
            lam_i = float(np.median(lam_local[s:e]))
            seg = x0[s:e] - b0[s:e]
            mask_i = None if not qrs_aware else base_mask[s:e]
            b1_seg = baseline_asls_masked(seg, lam=max(3e4, lam_i), p=asls_p,
                                          niter=10, mask=mask_i, decim_for_baseline=max(1, int(asls_decim)))
            b1[s:e] = b1_seg
        # 변화점 ±pad 초 로컬 리핏
        pad = int(round(break_pad_s * fs))
        if pad > 0:
            from scipy.ndimage import percentile_filter as _pf, median_filter as _mf
            for b in brks:
                lo = max(0, b - pad); hi = min(N, b + pad + 1)
                wloc = max(3, int(round(0.35 * fs))); wloc += (wloc % 2 == 0)
                resid = (x0[lo:hi] - b0[lo:hi] - b1[lo:hi])
                b_loc = _pf(resid, percentile=20, size=wloc, mode='nearest')
                b_loc = _mf(b_loc, size=max(3, int(round(0.12*fs))), mode='nearest')
                b1[lo:hi] += b_loc

    # 4) 매우저주파 안정화 + 잔차 클램프
    b = b0 + b1
    b_slow = _onepole(b, verylow_fc, fs)
    clamp_w = max(3, int(round(clamp_win_s * fs))); clamp_w += (clamp_w % 2 == 0)
    from scipy.ndimage import median_filter
    resid = x - b_slow
    off = median_filter(resid, size=clamp_w, mode='nearest')
    b_final = b_slow + off
    y_corr = x - b_final
    return y_corr, b_final

# Version2 wrapper (optimized variant from calibration_edit.py)
def baseline_hybrid_plus_adaptive_v2(
    y, fs,
    per_win_s=2.8, per_q=15,
    asls_lam=1e8, asls_p=0.01, asls_decim=12,
    qrs_aware=True, verylow_fc=0.03, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    hard_cut=True, break_pad_s=0.30
):
    if _baseline_v2 is None:
        return baseline_hybrid_plus_adaptive(
            y, fs,
            per_win_s=per_win_s, per_q=per_q,
            asls_lam=asls_lam, asls_p=asls_p, asls_decim=asls_decim,
            qrs_aware=qrs_aware, verylow_fc=verylow_fc, clamp_win_s=clamp_win_s,
            vol_win_s=vol_win_s, vol_gain=vol_gain, lam_floor_ratio=lam_floor_ratio,
            hard_cut=hard_cut, break_pad_s=break_pad_s
        )
    return _baseline_v2(
        y, fs,
        per_win_s=per_win_s, per_q=per_q,
        asls_lam=asls_lam, asls_p=asls_p, asls_decim=asls_decim,
        qrs_aware=qrs_aware, verylow_fc=verylow_fc, clamp_win_s=clamp_win_s,
        vol_win_s=vol_win_s, vol_gain=vol_gain, lam_floor_ratio=lam_floor_ratio,
        hard_cut=hard_cut, break_pad_s=break_pad_s
    )

def baseline_zero_drift(y, fs, cutoff=0.5, order=4):
    """
    간단·견고한 0-축 기준 보정기.
    - y_corr: high-pass로 드리프트 제거된 신호(평균≈0)
    - b_final: 추정된 드리프트(원본-보정)
    """
    y = np.asarray(y, float)
    y_corr = remove_baseline_drift(y, fs=fs, cutoff=float(cutoff), order=int(order))
    b_final = y - y_corr
    return y_corr, b_final

# 근본적 개선안: 2-타임스케일(빠른/느린) BL 추정 + 노이즈 가중 혼합
def baseline_improved(
    y, fs,
    per_win_s=2.8, per_q=15,
    asls_lam=1e8, asls_p=0.01, asls_decim=12,
    qrs_aware=True, verylow_fc=0.03, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    lam_slow_gain=10.0, hv_z_thr=1.5
):
    """
    아이디어
    - 빠른(중간 λ)과 느린(큰 λ) BL을 각각 추정 후, 변동성(z_vol)에 따라 가중 혼합.
    - 노이즈↑(z_vol↑)일 때 느린 BL 비중↑ → 기준축 흔들림 억제.
    - QRS/hardware noise 구간은 마스크로 피팅 영향 최소화.
    """
    from scipy.ndimage import percentile_filter, median_filter
    x = np.asarray(y, float); N = x.size
    if N < 8:
        return np.zeros_like(x), np.zeros_like(x)

    # 초기 바닥선
    w = max(3, int(round(per_win_s * fs)));  w += (w % 2 == 0)
    dc = np.median(x[np.isfinite(x)])
    x0 = x - dc
    b0 = percentile_filter(x0, percentile=per_q, size=w, mode='nearest')

    # QRS 보호 마스크
    try:
        base_mask = make_qrs_mask(x, fs=fs) if qrs_aware else np.ones_like(x, bool)
    except Exception:
        base_mask = np.ones_like(x, bool)

    # 변동성 측정 + HV 마스크(피팅 배제)
    vw = max(5, int(round(vol_win_s * fs)));  vw += (vw % 2 == 0)
    k = np.ones(vw, float)
    s1 = np.convolve(x, k, mode='same'); s2 = np.convolve(x*x, k, mode='same')
    m = s1 / vw; v = s2 / vw - m*m; v[v < 0] = 0.0
    rs = np.sqrt(v)
    rs_ref = np.percentile(rs, 90) + 1e-9
    z_vol = np.clip(rs / rs_ref, 0, 10.0)
    hv_mask = z_vol > float(hv_z_thr)
    mask_fit = base_mask & (~hv_mask)

    # 빠른/느린 BL 추정
    seg = x0 - b0
    lam_fast = float(max(5e4, asls_lam))
    lam_slow = float(max(5e4, asls_lam * float(lam_slow_gain)))
    b_fast = baseline_asls_masked(seg, lam=lam_fast, p=asls_p, niter=10,
                                  mask=mask_fit, decim_for_baseline=max(1, int(asls_decim)))
    b_slow = baseline_asls_masked(seg, lam=lam_slow, p=asls_p, niter=10,
                                  mask=mask_fit, decim_for_baseline=max(1, int(asls_decim)))

    # 혼합 가중: 노이즈↑ → 느린 BL 가중↑
    alpha_fast = 1.0 / (1.0 + float(vol_gain) * z_vol)  # 0~1
    alpha_fast = np.clip(alpha_fast, 0.0, 1.0)
    b1 = alpha_fast * b_fast + (1.0 - alpha_fast) * b_slow

    # 후처리(매우저주파 안정화 + 잔차 오프셋)
    b = b0 + b1
    b_slow2 = _onepole(b, verylow_fc, fs)
    clamp_w = max(3, int(round(clamp_win_s * fs))); clamp_w += (clamp_w % 2 == 0)
    resid = x - b_slow2
    off = median_filter(resid, size=clamp_w, mode='nearest')
    b_final = b_slow2 + off
    y_corr = x - b_final
    return y_corr, b_final
>>>>>>> Stashed changes

# =========================
# Residual-based selective refit
# =========================
def selective_residual_refit(y_src, base_in, fs,
                             k_sigma=3.2, win_s=0.5, pad_s=0.20,
                             method='percentile', per_q=20,
                             asls_lam=5e4, asls_p=0.01, asls_decim=6):
    from scipy.ndimage import median_filter, percentile_filter
    y_src = np.asarray(y_src, float); base = np.asarray(base_in, float).copy()
    N = y_src.size
    if N < 10: return (y_src - base), base, np.zeros(N, bool)

    resid = y_src - base
    med = np.median(resid); mad = np.median(np.abs(resid - med)) + 1e-12
    z = np.abs((resid - med) / (1.4826 * mad))

    cand = z > float(k_sigma)
    pad = int(round(pad_s * fs))
    if pad > 0:
        k = np.ones(pad*2+1, dtype=int)
        cand = (np.convolve(cand.astype(int), k, mode='same') > 0)

    wloc = max(3, int(round(win_s * fs)))
    if wloc % 2 == 0: wloc += 1
    refit_mask = np.zeros(N, bool)
    i = 0
    while i < N:
        if not cand[i]: i += 1; continue
        j = i
        while j < N and cand[j]: j += 1
        a, b = i, j
        if (b - a) >= max(5, int(0.20 * fs)):
            if method == 'percentile':
                aa = max(0, a - wloc//2); bb = min(N, b + wloc//2)
                seg_ctx = resid[aa:bb]
                loc = percentile_filter(seg_ctx, percentile=int(per_q), size=wloc, mode='nearest')
                loc = loc[(a-aa):(a-aa)+(b-a)]
                L = b - a
                if L > 8:
                    from scipy.ndimage import median_filter as _mf
                    taper = np.ones(L, float)
                    tlen = min(L//3, max(3, int(0.06*fs)))
                    if tlen > 0:
                        win = np.hanning(2*tlen)
                        taper[:tlen] = win[:tlen]
                        taper[-tlen:] = win[-tlen:]
                    loc = _mf(loc, size=max(3, int(0.10*fs)), mode='nearest') * taper
                base[a:b] += loc
            else:
                aa = max(0, a - wloc//2); bb = min(N, b + wloc//2)
                seg_ctx = y_src[aa:bb] - base[aa:bb]
                b_loc = baseline_asls_masked(seg_ctx, lam=float(asls_lam), p=float(asls_p),
                                             niter=8, mask=None,
                                             decim_for_baseline=max(1, int(asls_decim)))
                b_loc = b_loc[(a-aa):(a-aa)+(b-a)]
                L = b - a
                if L > 8:
                    from scipy.ndimage import median_filter as _mf
                    taper = np.ones(L, float)
                    tlen = min(L//3, max(3, int(0.06*fs)))
                    if tlen > 0:
                        win = np.hanning(2*tlen)
                        taper[:tlen] = win[:tlen]
                        taper[-tlen:] = win[-tlen:]
                    b_loc = _mf(b_loc, size=max(3, int(0.10*fs)), mode='nearest') * taper
                base[a:b] += b_loc
            refit_mask[a:b] = True
        i = j

    y_corr2 = y_src - base
    return y_corr2, base, refit_mask

# =========================
# Masks (computed on processed signal)
# =========================
def suppress_negative_sag(y, fs, win_sec=1.0, q_floor=20, k_neg=3.5,
                          min_dur_s=0.25, pad_s=0.25, protect_qrs=True):
    from scipy.ndimage import percentile_filter
    y = np.asarray(y, float); N = y.size
    w = max(3, int(round(win_sec * fs)));  w += (w % 2 == 0)
    floor  = percentile_filter(y, percentile=q_floor, size=w, mode='nearest')
    median = percentile_filter(y, percentile=50,     size=w, mode='nearest')
    r = y - median; neg = np.minimum(r, 0.0)
    med = np.median(neg); mad = np.median(np.abs(neg - med)) + 1e-12
    zneg = (neg - med) / (1.4826 * mad)
    mask = (zneg < -abs(k_neg)) & (y < floor)
    if protect_qrs:
        try:
            import neurokit2 as nk
            info = nk.ecg_peaks(y, sampling_rate=fs)[1]
            r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            prot = np.zeros(N, bool); pad = int(round(0.12 * fs))
            for r0 in r_idx:
                lo = max(0, r0 - pad); hi = min(N, r0 + pad + 1); prot[lo:hi] = True
            mask &= (~prot)
        except Exception: pass
    min_len = int(round(min_dur_s * fs)); pad_n = int(round(pad_s * fs))
    out = np.zeros(N, bool); i = 0
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]: j += 1
            if (j-i) >= min_len:
                lo = max(0, i - pad_n); hi = min(N, j + pad_n)
                out[lo:hi] = True
            i = j
        else: i += 1
    return out

def fix_downward_steps_mask(y, fs, pre_s=0.5, post_s=0.5, gap_s=0.08,
                            amp_sigma=5.0, amp_abs=None, min_hold_s=0.45,
                            refractory_s=0.80, protect_qrs=True):
    from scipy.ndimage import median_filter
    y = np.asarray(y, float); N = y.size
    if N < 10: return np.zeros(N, bool)
    qrs_prot = np.zeros(N, bool)
    if protect_qrs:
        try:
            import neurokit2 as nk
            info = nk.ecg_peaks(y, sampling_rate=fs)[1]
            r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            pad = int(round(0.12 * fs))
            for r in r_idx:
                lo = max(0, r - pad); hi = min(N, r + pad + 1); qrs_prot[lo:hi] = True
        except Exception: pass
    m_win = max(3, int(round(0.12 * fs)));  m_win += (m_win % 2 == 0)
    y_s = median_filter(y, size=m_win, mode='nearest')
    stride = max(1, int(round(0.02 * fs)))
    idxs = np.arange(0, N, stride)
    pre = int(round(pre_s * fs)); post = int(round(post_s * fs))
    gap = int(round(gap_s * fs));  hold = int(round(min_hold_s * fs))
    refr = int(round(refractory_s * fs))
    med = np.median(y_s); mad = np.median(np.abs(y_s - med)) + 1e-12
    thr = amp_sigma * 1.4826 * mad
    if amp_abs is not None: thr = max(thr, float(amp_abs))
    mask = np.zeros(N, bool); last_end = -10**9
    for i in idxs:
        if i - last_end < refr or qrs_prot[i]: continue
        a = max(0, i - pre); b = min(N, i + gap); c = min(N, i + gap + post)
        if (b-a) < int(0.2*fs) or (c-b) < int(0.2*fs): continue
        m1 = np.median(y_s[a:b]); m2 = np.median(y_s[b:c]); drop = m1 - m2
        if drop < thr: continue
        hold_end = min(N, c + hold); m_hold = np.median(y_s[c:hold_end]) if c < hold_end else m2
        if (m1 - m_hold) < 0.6 * drop: continue
        mask[c:hold_end] = True
        last_end = hold_end
    return mask

def smooth_corners_mask(y, fs, L_ms=140, k_sigma=5.5, protect_qrs=True):
    y = np.asarray(y, float); N = y.size
    if N < 10: return np.zeros(N, bool)
    d1 = np.diff(y, prepend=y[0]); d2 = np.diff(d1, prepend=d1[0])
    med = np.median(d2); mad = np.median(np.abs(d2 - med)) + 1e-12
    z = (d2 - med) / (1.4826 * mad)
    cand = np.abs(z) > float(k_sigma)
    if protect_qrs:
        try:
            import neurokit2 as nk
            r_idx = np.array(nk.ecg_peaks(y, sampling_rate=fs)[1].get("ECG_R_Peaks", []), dtype=int)
            pad = int(round(0.12 * fs))
            prot = np.zeros(N, dtype=bool)
            for r in r_idx:
                lo = max(0, r - pad); hi = min(N, r + pad + 1); prot[lo:hi] = True
            cand &= (~prot)
        except Exception: pass
    idx = np.flatnonzero(cand)
    if idx.size == 0: return np.zeros(N, bool)
    L = max(3, int(round(L_ms * 1e-3 * fs)))
    keep = []; last = -10**9
    for i in idx:
        if i - last > L: keep.append(i); last = i
    mask = np.zeros(N, bool)
    for i in keep:
        a = max(0, i - L); b = min(N, i + L)
        mask[a:b] = True
    return mask

def rolling_std_fast(y: np.ndarray, w: int) -> np.ndarray:
    y = y.astype(float); k = np.ones(int(w), float)
    s1 = np.convolve(y, k, mode='same'); s2 = np.convolve(y*y, k, mode='same')
    m = s1 / int(w); v = s2 / w - m*m; v[v < 0] = 0.0
    return np.sqrt(v)

def high_variance_mask(y: np.ndarray, win=2000, k_sigma=5.0, pad=125):
    x = y.astype(float)
    rs = rolling_std_fast(x, max(2,int(win)))
    rs_med = np.median(rs); rs_mad = np.median(np.abs(rs - rs_med)) + 1e-12
    thr = rs_med + 1.4826 * rs_mad * float(k_sigma)
    mask = rs > thr
    if pad and pad > 0:
        padk = np.ones(int(pad)*2+1, dtype=int)
        mask = (np.convolve(mask.astype(int), padk, mode='same') > 0)
    stats = {
        "threshold": float(thr),
        "removed_samples": int(mask.sum()),
        "kept_samples": int((~mask).sum()),
        "compression_ratio": float((~mask).sum()/y.size)
    }
    return mask, stats

def _smooth_binary(mask: np.ndarray, fs: float, blend_ms: int = 80) -> np.ndarray:
    L = max(3, int(round(blend_ms/1000.0 * fs)))
    if L % 2 == 0: L += 1
    win = np.hanning(L); win = win / win.sum()
    return np.convolve(mask.astype(float), win, mode='same')

def qrs_aware_wavelet_denoise(y, fs, wavelet='db6', level=None, sigma_scale=2.8, blend_ms=80):
    y = np.asarray(y, float); N = y.size
    try:
        mask = make_qrs_mask(y, fs=fs)
    except Exception:
        mask = np.ones_like(y, dtype=bool)
    alpha = _smooth_binary(mask, fs, blend_ms=blend_ms)
    try:
        import pywt
        if level is None:
            level = min(5, max(2, int(np.log2(fs/8.0))))
        coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode='symmetric')
        cA, details = coeffs[0], coeffs[1:]
        sigma = np.median(np.abs(details[-1])) / 0.6745 + 1e-12
        thr = float(sigma_scale) * sigma
        details_d = [pywt.threshold(c, thr, mode='soft') for c in details]
        y_w = pywt.waverec([cA] + details_d, wavelet=wavelet, mode='symmetric')
        if y_w.size != N: y_w = y_w[:N]
    except Exception:
        from scipy.signal import savgol_filter
        win = max(5, int(round(0.05 * fs)));  win += (win % 2 == 0)
        y_w = savgol_filter(y, window_length=win, polyorder=2, mode='interp')
    return alpha * y_w + (1.0 - alpha) * y, alpha

def burst_mask(y, fs, win_ms=140, k_diff=7.5, k_std=3.5, pad_ms=80, protect_qrs=True):
    """버스트성 급변/분산 상승 마스크"""
    y = np.asarray(y, float); N = y.size
    if N < 10: return np.zeros(N, bool)
    dy = np.diff(y, prepend=y[0])
    dmed = np.median(dy); dmad = np.median(np.abs(dy - dmed)) + 1e-12
    z_diff = (dy - dmed) / (1.4826 * dmad)

    w = max(3, int(round((win_ms/1000.0) * fs)));  w += (w % 2 == 0)
    k = np.ones(w, float)
    s1 = np.convolve(y, k, mode='same'); s2 = np.convolve(y*y, k, mode='same')
    m = s1 / w; v = s2 / w - m*m; v[v < 0] = 0.0
    rs = np.sqrt(v)
    rmed = np.median(rs); rmad = np.median(np.abs(rs - rmed)) + 1e-12
    z_std = (rs - rmed) / (1.4826 * rmad)

    cand = (np.abs(z_diff) > float(k_diff)) & (z_std > float(k_std))
    if protect_qrs:
        try:
            import neurokit2 as nk
            r_idx = np.array(nk.ecg_peaks(y, sampling_rate=fs)[1].get("ECG_R_Peaks", []), dtype=int)
            prot = np.zeros(N, dtype=bool); pad_r = int(round(0.12 * fs))
            for r in r_idx:
                lo = max(0, r - pad_r); hi = min(N, r + pad_r + 1); prot[lo:hi] = True
            cand &= (~prot)
        except Exception:
            pass
    pad = int(round((pad_ms/1000.0) * fs))
    if pad > 0:
        padk = np.ones(pad*2+1, dtype=int)
        cand = (np.convolve(cand.astype(int), padk, mode='same') > 0)
    return cand

# =========================
# Custom X-only stretch zoom ViewBox (Shift+좌클릭 드래그)
# =========================
class XZoomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self.setMouseEnabled(x=True, y=True)
        self.setLimits(yMin=-1e12, yMax=1e12)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton and (ev.modifiers() & QtCore.Qt.ShiftModifier):
            ev.accept()
            pos = ev.pos()
            last = ev.lastPos()
            dx = pos.x() - last.x()
            s = np.exp(-dx * 0.005)
            s = float(np.clip(s, 1e-3, 1e3))
            center = self.mapSceneToView(pos)
            self.scaleBy((s, 1.0), center=center)  # X만 확대/축소
        else:
            super().mouseDragEvent(ev, axis=axis)

# =========================
# Qt Viewer
# =========================
class ECGViewer(QtWidgets.QWidget):
    def __init__(self, t, y_raw, parent=None):
        super().__init__(parent)
        self.t = t; self.y_raw = y_raw
        self._recompute_timer = None

        root = QtWidgets.QVBoxLayout(self)

        # ====== View Toggles ======
        tg = QtWidgets.QHBoxLayout()
        self.cb_raw   = QtWidgets.QCheckBox("원본 신호");       self.cb_raw.setChecked(True)
        self.cb_corr  = QtWidgets.QCheckBox("가공(보정) 신호"); self.cb_corr.setChecked(True)
        self.cb_mask  = QtWidgets.QCheckBox("마스크 패널");     self.cb_mask.setChecked(True)
        self.cb_base  = QtWidgets.QCheckBox("Baseline 표시");   self.cb_base.setChecked(False)
        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            tg.addWidget(cb)
        tg.addStretch(1)
        root.addLayout(tg)

        # ====== Baseline (Hybrid BL++ only) ======
        bl = QtWidgets.QHBoxLayout()
        self.cb_qrsaware = QtWidgets.QCheckBox("QRS-aware"); self.cb_qrsaware.setChecked(True)
        self.sb_asls_lam = QtWidgets.QDoubleSpinBox(); self.sb_asls_lam.setRange(0.0,1e9); self.sb_asls_lam.setDecimals(0); self.sb_asls_lam.setValue(8e7)
        self.sb_asls_p = QtWidgets.QDoubleSpinBox();   self.sb_asls_p.setRange(0.001,0.2); self.sb_asls_p.setSingleStep(0.001); self.sb_asls_p.setValue(0.01)
        self.sb_per_win = QtWidgets.QDoubleSpinBox();  self.sb_per_win.setRange(0.5,10.0); self.sb_per_win.setValue(2.8); self.sb_per_win.setSingleStep(0.1)
        self.sb_per_q = QtWidgets.QSpinBox();          self.sb_per_q.setRange(1,49); self.sb_per_q.setValue(15)
        self.sb_asls_decim = QtWidgets.QSpinBox();     self.sb_asls_decim.setRange(1,64); self.sb_asls_decim.setValue(12)
        self.sb_lpf_fc = QtWidgets.QDoubleSpinBox();   self.sb_lpf_fc.setRange(0.005,0.5); self.sb_lpf_fc.setSingleStep(0.005); self.sb_lpf_fc.setValue(0.03)
        self.sb_vol_win = QtWidgets.QDoubleSpinBox();  self.sb_vol_win.setRange(0.1,5.0); self.sb_vol_win.setSingleStep(0.05); self.sb_vol_win.setValue(0.6)
        self.sb_vol_gain = QtWidgets.QDoubleSpinBox(); self.sb_vol_gain.setRange(0.1,50.0); self.sb_vol_gain.setSingleStep(0.1); self.sb_vol_gain.setValue(6.0)
        self.sb_lam_floor = QtWidgets.QDoubleSpinBox(); self.sb_lam_floor.setRange(0.1,50.0); self.sb_lam_floor.setSingleStep(0.1); self.sb_lam_floor.setValue(3.0)
        self.cb_break_cut = QtWidgets.QCheckBox("Hard cut at breaks"); self.cb_break_cut.setChecked(True)
        self.sb_break_pad = QtWidgets.QDoubleSpinBox();  self.sb_break_pad.setRange(0.0, 2.0); self.sb_break_pad.setSingleStep(0.05); self.sb_break_pad.setValue(0.30)
        self.cb_res_refit = QtWidgets.QCheckBox("Residual refit"); self.cb_res_refit.setChecked(True)
        self.cmb_res_method = QtWidgets.QComboBox(); self.cmb_res_method.addItems(["percentile", "asls"])
        self.sb_res_k = QtWidgets.QDoubleSpinBox(); self.sb_res_k.setRange(1.0, 10.0); self.sb_res_k.setSingleStep(0.1); self.sb_res_k.setValue(3.2)
        self.sb_res_win = QtWidgets.QDoubleSpinBox(); self.sb_res_win.setRange(0.05, 3.0); self.sb_res_win.setSingleStep(0.05); self.sb_res_win.setValue(0.5)
        self.sb_res_pad = QtWidgets.QDoubleSpinBox(); self.sb_res_pad.setRange(0.0, 1.5); self.sb_res_pad.setSingleStep(0.05); self.sb_res_pad.setValue(0.20)

        for lbl, w in [
            ("QRS", self.cb_qrsaware),
            ("λ", self.sb_asls_lam), ("p", self.sb_asls_p),
            ("PerWin(s)", self.sb_per_win), ("PerQ", self.sb_per_q),
            ("AsLS decim", self.sb_asls_decim), ("LPF fc", self.sb_lpf_fc),
            ("VOL win(s)", self.sb_vol_win), ("VOL gain", self.sb_vol_gain), ("λ floor(%)", self.sb_lam_floor),
            ("Break pad(s)", self.sb_break_pad), ("", self.cb_break_cut),
            ("Residual refit", self.cb_res_refit), ("mode", self.cmb_res_method),
            ("kσ", self.sb_res_k), ("win(s)", self.sb_res_win), ("pad(s)", self.sb_res_pad),
        ]:
            bl.addWidget(QtWidgets.QLabel(lbl) if lbl else QtWidgets.QLabel()); bl.addWidget(w)
        root.addLayout(bl)

        # ====== Mask params ======
        row2 = QtWidgets.QHBoxLayout()
        self.cb_sag = QtWidgets.QCheckBox("Sag"); self.cb_sag.setChecked(True)
        self.sb_sag_win = QtWidgets.QDoubleSpinBox(); self.sb_sag_win.setRange(0.2,5.0); self.sb_sag_win.setValue(1.0)
        self.sb_sag_q = QtWidgets.QSpinBox(); self.sb_sag_q.setRange(1,49); self.sb_sag_q.setValue(20)
        self.sb_sag_k = QtWidgets.QDoubleSpinBox(); self.sb_sag_k.setRange(0.5,10.0); self.sb_sag_k.setValue(3.5)
        self.sb_sag_mindur = QtWidgets.QDoubleSpinBox(); self.sb_sag_mindur.setRange(0.05,2.0); self.sb_sag_mindur.setValue(0.25)
        self.sb_sag_pad = QtWidgets.QDoubleSpinBox(); self.sb_sag_pad.setRange(0.0,1.0); self.sb_sag_pad.setValue(0.25)

        self.cb_step = QtWidgets.QCheckBox("Step"); self.cb_step.setChecked(True)
        self.sb_step_sigma = QtWidgets.QDoubleSpinBox(); self.sb_step_sigma.setRange(1.0,15.0); self.sb_step_sigma.setValue(5.0)
        self.sb_step_abs = QtWidgets.QDoubleSpinBox(); self.sb_step_abs.setRange(0.0,500.0); self.sb_step_abs.setValue(0.0)
        self.sb_step_hold = QtWidgets.QDoubleSpinBox(); self.sb_step_hold.setRange(0.1,2.0); self.sb_step_hold.setValue(0.45)

        self.cb_corner = QtWidgets.QCheckBox("Corner"); self.cb_corner.setChecked(True)
        self.sb_corner_L = QtWidgets.QSpinBox(); self.sb_corner_L.setRange(20,400); self.sb_corner_L.setValue(140)
        self.sb_corner_k = QtWidgets.QDoubleSpinBox(); self.sb_corner_k.setRange(1.0,15.0); self.sb_corner_k.setValue(5.5)

        self.cb_burst = QtWidgets.QCheckBox("Burst"); self.cb_burst.setChecked(True)
        self.sb_burst_win = QtWidgets.QSpinBox(); self.sb_burst_win.setRange(20,400); self.sb_burst_win.setValue(140)
        self.sb_burst_kd = QtWidgets.QDoubleSpinBox(); self.sb_burst_kd.setRange(1.0,20.0); self.sb_burst_kd.setValue(7.5)
        self.sb_burst_ks = QtWidgets.QDoubleSpinBox(); self.sb_burst_ks.setRange(1.0,20.0); self.sb_burst_ks.setValue(3.5)
        self.sb_burst_pad = QtWidgets.QSpinBox(); self.sb_burst_pad.setRange(0,400); self.sb_burst_pad.setValue(80)

        self.cb_wave = QtWidgets.QCheckBox("Wavelet"); self.cb_wave.setChecked(False)
        self.sb_wave_sigma = QtWidgets.QDoubleSpinBox(); self.sb_wave_sigma.setRange(1.0,6.0); self.sb_wave_sigma.setValue(2.8)
        self.sb_wave_blend = QtWidgets.QSpinBox(); self.sb_wave_blend.setRange(20,200); self.sb_wave_blend.setValue(80)

        self.win_sb = QtWidgets.QSpinBox(); self.win_sb.setRange(10,50000); self.win_sb.setValue(2000)
        self.kd_sb = QtWidgets.QDoubleSpinBox(); self.kd_sb.setRange(0.1,50.0); self.kd_sb.setSingleStep(0.1); self.kd_sb.setValue(5.0)
        self.pad_sb = QtWidgets.QSpinBox(); self.pad_sb.setRange(0,20000); self.pad_sb.setValue(125)

        for lbl, w in [
            ("Sag", self.cb_sag), ("Win(s)", self.sb_sag_win), ("q", self.sb_sag_q), ("k", self.sb_sag_k),
            ("minDur(s)", self.sb_sag_mindur), ("Pad(s)", self.sb_sag_pad),
            ("Step", self.cb_step), ("σ", self.sb_step_sigma), ("Abs", self.sb_step_abs), ("Hold(s)", self.sb_step_hold),
            ("Corner", self.cb_corner), ("L(ms)", self.sb_corner_L), ("kσ", self.sb_corner_k),
            ("Burst", self.cb_burst), ("Win(ms)", self.sb_burst_win), ("kΔ", self.sb_burst_kd), ("kstd", self.sb_burst_ks), ("Pad(ms)", self.sb_burst_pad),
            ("Wave", self.cb_wave), ("σ", self.sb_wave_sigma), ("Blend(ms)", self.sb_wave_blend),
            ("HV WIN", self.win_sb), ("HV Kσ", self.kd_sb), ("HV PAD", self.pad_sb),
        ]:
            row2.addWidget(QtWidgets.QLabel(lbl)); row2.addWidget(w)
        root.addLayout(row2)

        # ====== Y축 수동/자동 제어 UI ======
        yctl = QtWidgets.QHBoxLayout()
        self.btn_auto_y = QtWidgets.QPushButton("Auto Y-Scale: ON")
        self.btn_auto_y.setCheckable(True); self.btn_auto_y.setChecked(True)
        self.ymin_spin = QtWidgets.QDoubleSpinBox(); self.ymin_spin.setRange(-1e6, 1e6); self.ymin_spin.setDecimals(6); self.ymin_spin.setValue(-1.0)
        self.ymax_spin = QtWidgets.QDoubleSpinBox(); self.ymax_spin.setRange(-1e6, 1e6); self.ymax_spin.setDecimals(6); self.ymax_spin.setValue(1.0)
        self.ymin_spin.setEnabled(False); self.ymax_spin.setEnabled(False)
        yctl.addWidget(self.btn_auto_y)
        yctl.addWidget(QtWidgets.QLabel("Ymin")); yctl.addWidget(self.ymin_spin)
        yctl.addWidget(QtWidgets.QLabel("Ymax")); yctl.addWidget(self.ymax_spin)
        yctl.addStretch(1)
        root.addLayout(yctl)

        # ====== Plots ======
        self.win_plot = pg.GraphicsLayoutWidget(); root.addWidget(self.win_plot)

        self.plot = self.win_plot.addPlot(row=0, col=0, viewBox=XZoomViewBox())
        self.plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot.setLabel('bottom','Time (s)'); self.plot.setLabel('left','Amplitude')
        self.plot.showGrid(x=True,y=True,alpha=0.3)

        self.overview = self.win_plot.addPlot(row=1, col=0); self.overview.setMaximumHeight(150); self.overview.showGrid(x=True,y=True,alpha=0.2)
        self.region = pg.LinearRegionItem(); self.region.setZValue(10); self.overview.addItem(self.region); self.region.sigRegionChanged.connect(self.update_region)

        # Colors: raw(gray), corrected(yellow), baseline(cyan dashed)
        pen_raw  = pg.mkPen(color=(150, 150, 150), width=1)
        pen_corr = pg.mkPen(color=(255, 215, 0),   width=1.6)  # Yellow (Gold)
        pen_base = pg.mkPen(color=(0, 200, 255),   width=1, style=QtCore.Qt.DashLine)

        self.curve_raw  = self.plot.plot([], [], pen=pen_raw)
        self.curve_corr = self.plot.plot([], [], pen=pen_corr)
        self.curve_base = self.plot.plot([], [], pen=pen_base); self.curve_base.setVisible(False)
        self.curve_corr.setZValue(5); self.curve_raw.setZValue(3); self.curve_base.setZValue(2)

        self.ov_curve = self.overview.plot([], [], pen=pg.mkPen(width=1))

        self.mask_plot = self.win_plot.addPlot(row=2, col=0); self.mask_plot.setMaximumHeight(130)
        self.mask_plot.setLabel('left','Masks'); self.mask_plot.setLabel('bottom','Time (s)')
        self.mask_plot.showGrid(x=True,y=True,alpha=0.2)
        self.hv_curve    = self.mask_plot.plot([], [], pen=pg.mkPen(width=1))
        self.sag_curve   = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DotLine))
        self.step_curve  = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashLine))
        self.corner_curve= self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.SolidLine))
        self.burst_curve = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashDotLine))
        self.wave_curve  = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashDotDotLine))
        self.resrefit_curve = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashLine))

        # ---- 이벤트 연결 ----
        def connect_change(w, slot):
            if isinstance(w, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                w.valueChanged.connect(slot)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.toggled.connect(slot)
            elif isinstance(w, QtWidgets.QPushButton):
                w.clicked.connect(slot)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(lambda _=None: slot())

        for w in [
            self.cb_qrsaware, self.sb_asls_lam, self.sb_asls_p, self.sb_per_win, self.sb_per_q,
            self.sb_asls_decim, self.sb_lpf_fc,
            self.sb_vol_win, self.sb_vol_gain, self.sb_lam_floor,
            self.cb_break_cut, self.sb_break_pad,
            self.cb_res_refit, self.cmb_res_method, self.sb_res_k, self.sb_res_win, self.sb_res_pad,
            self.cb_sag, self.sb_sag_win, self.sb_sag_q, self.sb_sag_k, self.sb_sag_mindur, self.sb_sag_pad,
            self.cb_step, self.sb_step_sigma, self.sb_step_abs, self.sb_step_hold,
            self.cb_corner, self.sb_corner_L, self.sb_corner_k,
            self.cb_burst, self.sb_burst_win, self.sb_burst_kd, self.sb_burst_ks, self.sb_burst_pad,
            self.cb_wave, self.sb_wave_sigma, self.sb_wave_blend,
            self.win_sb, self.kd_sb, self.pad_sb,
        ]:
            connect_change(w, self.schedule_recompute)

        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            cb.toggled.connect(self.update_visibility)

        # Y축 제어 핸들러
        self.btn_auto_y.clicked.connect(self._toggle_y_auto)
        self.ymin_spin.valueChanged.connect(self._apply_y_range_from_spins)
        self.ymax_spin.valueChanged.connect(self._apply_y_range_from_spins)

        # Data
        self.set_data(t, y_raw)

        def dblclick(ev):
            if ev.double():
                self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot.scene().sigMouseClicked.connect(dblclick)

    # ---- Y축 제어 메서드 ----
    def _toggle_y_auto(self):
        state = self.btn_auto_y.isChecked()
        self.plot.enableAutoRange('y', state)
        self.btn_auto_y.setText(f"Auto Y-Scale: {'ON' if state else 'OFF'}")
        self.ymin_spin.setEnabled(not state)
        self.ymax_spin.setEnabled(not state)
        if not state:
            self._apply_y_range_from_spins()

    def _apply_y_range_from_spins(self):
        if self.btn_auto_y.isChecked():
            return
        ylo = self.ymin_spin.value()
        yhi = self.ymax_spin.value()
        if yhi <= ylo:
            yhi = ylo + 1e-9
            self.ymax_spin.setValue(yhi)
        self.plot.setYRange(ylo, yhi, padding=0)

    # ---- 디바운스 재계산 ----
    def schedule_recompute(self):
        if self._recompute_timer is None:
            self._recompute_timer = QtCore.QTimer(self)
            self._recompute_timer.setSingleShot(True)
            self._recompute_timer.timeout.connect(self.recompute)
        self._recompute_timer.start(600)

    def set_data(self, t, y):
        # 평균 제거(0 기준 중심화)
        y_centered = np.asarray(y, float)
        if y_centered.size > 0:
            y_centered = y_centered - float(np.nanmean(y_centered))

        self.t = np.asarray(t, float)
        self.y_raw = y_centered

        # 플롯 초기 세팅
        self.curve_raw.setData(self.t, self.y_raw)
        self.ov_curve.setData(self.t, self.y_raw)

        # 초기 영역
        end_t = min(self.t[0]+40.0, self.t[-1]) if self.t.size>1 else 0.0
        self.region.setRegion([self.t[0], end_t])

        # Y 스핀 초기값 갱신(데이터 기반)
        if self.y_raw.size > 0:
            y_min, y_max = float(np.min(self.y_raw)), float(np.max(self.y_raw))
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                m = 0.1 * (y_max - y_min)
                self.ymin_spin.blockSignals(True); self.ymax_spin.blockSignals(True)
                self.ymin_spin.setValue(y_min - m)
                self.ymax_spin.setValue(y_max + m)
                self.ymin_spin.blockSignals(False); self.ymax_spin.blockSignals(False)

        self.recompute()

    def update_visibility(self):
        self.curve_raw.setVisible(self.cb_raw.isChecked())
        self.curve_corr.setVisible(self.cb_corr.isChecked())
        self.mask_plot.setVisible(self.cb_mask.isChecked())
        self.curve_base.setVisible(self.cb_base.isChecked())

<<<<<<< Updated upstream
    def recompute(self):
        # 1) Baseline — Hybrid BL++
        y_src = self.y_raw.copy()
        y_corr, base = baseline_hybrid_plus_adaptive(
            y_src, FS,
            per_win_s=float(self.sb_per_win.value()),
            per_q=int(self.sb_per_q.value()),
            asls_lam=float(self.sb_asls_lam.value()),
            asls_p=float(self.sb_asls_p.value()),
            asls_decim=int(self.sb_asls_decim.value()),
            qrs_aware=self.cb_qrsaware.isChecked(),
            verylow_fc=float(self.sb_lpf_fc.value()),
            clamp_win_s=6.0,
            vol_win_s=float(self.sb_vol_win.value()),
            vol_gain=float(self.sb_vol_gain.value()),
            lam_floor_ratio=float(self.sb_lam_floor.value())/100.0,
            hard_cut=self.cb_break_cut.isChecked(),
            break_pad_s=float(self.sb_break_pad.value())
        )

        # 1.5) Residual selective refit
        resrefit_mask = np.zeros_like(y_corr, dtype=bool)
        if self.cb_res_refit.isChecked():
            mode = self.cmb_res_method.currentText()
            y_corr2, base2, resrefit_mask = selective_residual_refit(
                y_src, base, FS,
                k_sigma=float(self.sb_res_k.value()),
                win_s=float(self.sb_res_win.value()),
                pad_s=float(self.sb_res_pad.value()),
                method=mode,
                per_q=20, asls_lam=5e4, asls_p=0.01, asls_decim=6
            )
            y_corr, base = y_corr2, base2

        # === No AGC / No Glitch ===
        y_corr_eq = y_corr  # 처리 신호는 순수 BL++(+선택적 리핏) 결과
=======
    def recompute(self):
        # 1) Strict zero-baseline correction
        #    - corrected는 항상 0선 주위 진동
        #    - baseline(하늘색)은 드리프트 추정치
        #    - 가공 baseline(주황)은 0선(고정 축)
        y_src = self.y_raw.copy()
        y_corr_eq = remove_baseline_drift(y_src, fs=FS, cutoff=0.5, order=4)
        base     = y_src - y_corr_eq
        base_proc = np.zeros_like(base)
        resrefit_mask = np.zeros_like(y_corr_eq, dtype=bool)
>>>>>>> Stashed changes

        # 2) Masks on processed signal
        sag_mask = suppress_negative_sag(
            y_corr_eq, FS, win_sec=float(self.sb_sag_win.value()), q_floor=int(self.sb_sag_q.value()),
            k_neg=float(self.sb_sag_k.value()), min_dur_s=float(self.sb_sag_mindur.value()),
            pad_s=float(self.sb_sag_pad.value()), protect_qrs=True) if self.cb_sag.isChecked() else np.zeros_like(y_corr_eq, bool)

        step_mask = fix_downward_steps_mask(
            y_corr_eq, FS, amp_sigma=float(self.sb_step_sigma.value()),
            amp_abs=(None if float(self.sb_step_abs.value()) <= 0 else float(self.sb_step_abs.value())),
            min_hold_s=float(self.sb_step_hold.value()),
            protect_qrs=True) if self.cb_step.isChecked() else np.zeros_like(y_corr_eq, bool)

        corner_mask = smooth_corners_mask(
            y_corr_eq, FS, L_ms=int(self.sb_corner_L.value()),
            k_sigma=float(self.sb_corner_k.value()), protect_qrs=True) if self.cb_corner.isChecked() else np.zeros_like(y_corr_eq, bool)

        b_mask = np.zeros_like(y_corr_eq, bool)
        if self.cb_burst.isChecked():
            b_mask = burst_mask(
                y_corr_eq, FS, win_ms=int(self.sb_burst_win.value()),
                k_diff=float(self.sb_burst_kd.value()), k_std=float(self.sb_burst_ks.value()),
                pad_ms=int(self.sb_burst_pad.value()), protect_qrs=True)

        alpha_w = np.zeros_like(y_corr_eq)
        if self.cb_wave.isChecked():
            _, alpha_w = qrs_aware_wavelet_denoise(
                y_corr_eq, FS, sigma_scale=float(self.sb_wave_sigma.value()),
                blend_ms=int(self.sb_wave_blend.value()))

        hv_mask, hv_stats = high_variance_mask(
            y_corr_eq, win=int(self.win_sb.value()),
            k_sigma=float(self.kd_sb.value()), pad=int(self.pad_sb.value()))

        # 표시 업데이트
        self.curve_base.setData(self.t, base)
        self.curve_corr.setData(self.t, y_corr_eq)
        self.curve_raw.setData(self.t, self.y_raw)

        # 마스크 패널
        self.hv_curve.setData(self.t, hv_mask.astype(int))
        self.sag_curve.setData(self.t, sag_mask.astype(int))
        self.step_curve.setData(self.t, step_mask.astype(int))
        self.corner_curve.setData(self.t, corner_mask.astype(int))
        self.burst_curve.setData(self.t, b_mask.astype(int))
        self.wave_curve.setData(self.t, (alpha_w > 0.5).astype(int))
        self.resrefit_curve.setData(self.t, resrefit_mask.astype(int))

        txt = (
            f"HV removed={int(hv_mask.sum())} ({100*hv_mask.mean():.2f}%) | "
            f"kept={len(y_corr_eq)-int(hv_mask.sum())} | ratio={(1-hv_mask.mean()):.3f}"
        )
        self.mask_plot.setTitle(txt)

        self.update_visibility()

        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)

        # 자동 Y스케일이면 가시 구간 기반으로 margin 포함하여 설정
        if self.btn_auto_y.isChecked():
            vis_idx = (self.t >= lo) & (self.t <= hi)
            if np.any(vis_idx):
                y_sub = self.y_raw[vis_idx]
                ymin, ymax = float(np.min(y_sub)), float(np.max(y_sub))
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    margin = 0.1 * (ymax - ymin) if (ymax - ymin) > 0 else 1.0
                    self.plot.setYRange(ymin - margin, ymax + margin, padding=0)

    def update_region(self):
        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)

        if self.btn_auto_y.isChecked():
            vis_idx = (self.t >= lo) & (self.t <= hi)
            if np.any(vis_idx):
                y_sub = self.y_raw[vis_idx]
                y_min, y_max = np.min(y_sub), np.max(y_sub)
                if np.isfinite(y_min) and np.isfinite(y_max) and (y_max > y_min):
                    margin = 0.1 * (y_max - y_min)
                    self.plot.setYRange(float(y_min - margin), float(y_max + margin), padding=0)

# =========================
# Main
# =========================
def main():
    with FILE_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    ecg_raw = extract_ecg(data); assert ecg_raw is not None and ecg_raw.size > 0
    ecg = decimate_if_needed(ecg_raw, DECIM)
    t = np.arange(ecg.size) / FS

    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle('Fusion')
    w = QtWidgets.QMainWindow()
    viewer = ECGViewer(t, ecg)
    w.setWindowTitle(f"ECG Viewer — {int(FS_RAW)}→{int(FS)} Hz | Hybrid BL++ (AGC/Glitch 없음) | Masks on processed signal | No interpolation")
    w.setCentralWidget(viewer); w.resize(1480, 930); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
