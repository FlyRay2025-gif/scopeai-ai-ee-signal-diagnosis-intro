"""
第 2 步：重写的特征（features_v2）
===================================

和作者那版（01-readings/scopeai/features.py）的区别，一句话：
**尽量让每个特征无量纲，并且在"没有信号"时不失效。**

输入：一维电压序列 + 采样率
输出：10 个标量（下表的 key）

| key            | 含义                     | 为什么这么定                                   |
|----------------|--------------------------|-----------------------------------------------|
| vmean          | 直流电平 (V)              | 555 无定时电容时输出"卡在高电平"、没供电时"卡在 0 V"——去均值会把这区分扔掉 |
| vpp_v          | 峰峰值 (V)                | 代替 ptp/2：不截断、不除二，0–5 V 电路上记 5.0  |
| rms_v          | 交流有效值 (V)            | 功率视角；注意它会随占空比塌，别单独当"幅度"用   |
| crest          | 峰值 / 交流有效值         | 方波≈1.3、正弦≈1.41；但窄脉冲也会抬高它         |
| duty           | 占空比（1%–99% 分位中点） | 555 的 R2 会改变占空比，v1 完全没有这个量        |
| freq_hz        | 基频（抛物线插值）         | 比 argmax 少一个 bin 的量化误差                 |
| flatness       | 谱平坦度                  | 白噪声≈0.56，单频≈0.01；这是真正的"噪声探测器"   |
| spread_rel     | 谱宽 / 基频               | 只在显著谱线上算，否则噪声底噪会把值拉爆         |
| zcr_per_cycle  | 每周期过零次数             | 干净周期信号≈1，带毛刺>1；噪声时必须靠 flatness 兜底 |
| jitter_rel     | 上升过零间隔的变异系数 CV | 代替 ms 抖动：1 Hz 和 1 kHz 可比                |

关于"过零率"：它的量纲是 Hz（次/秒）。不做归一化时，噪声恒为 fs/2；
除以 2*f0 之后才是"每周期过几次零"。但注意：f0 在纯噪声下是从噪声里挑出来的
随机值，所以这个比值**分不开噪声和信号**——实测干净方波 0.97，纯噪声 0.75–2.07。
噪声判断交给 flatness，这个比值只用来描述"有基频的信号有多少毛刺"。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

FEATURES = [
    "vmean",
    "vpp_v",
    "rms_v",
    "crest",
    "duty",
    "freq_hz",
    "rise_ms",
    "flatness",
    "spread_rel",
    "zcr_per_cycle",
    "jitter_rel",
]

# 低于这个有效值就当作"没有信号"，比值类特征直接给 0，避免 0/0 炸出天文数字
SILENT_RMS = 0.02


def _fundamental(xc: np.ndarray, fs: int) -> float:
    """基频：除直流外最大谱峰 + 抛物线插值（把 bin 量化误差再压一半）。"""
    n = xc.size
    mags = np.abs(np.fft.rfft(xc))
    if mags.size < 4:
        return 0.0
    k = int(np.argmax(mags[1:]) + 1)
    if mags[k] < 0.2 * np.percentile(mags[1:], 95):
        return 0.0                      # 没有明显谱峰 = 没有基频
    # 用左右两个邻点做抛物线顶点插值
    a, b, c = mags[k - 1], mags[k], mags[k + 1] if k + 1 < mags.size else mags[k]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return (k + delta) * fs / n


def _rise_ms(x: np.ndarray, fs: int) -> float:
    """上升时间：从低电平 10% 爬到 90% 所用的毫秒数（取多次上升沿的中位数）。

    谱宽看不出边沿细节——RC 的 τ 从 1 ms 变到 0.02 ms，谐波结构几乎不变，
    但上升时间会从 2.2 ms 塌到 0.1 ms。这是"换一个物理量才测得出来"的典型例子。
    """
    hi, lo = float(np.percentile(x, 99)), float(np.percentile(x, 1))
    span = hi - lo
    if span < 0.05:
        return 0.0
    t10, t90 = lo + 0.1 * span, lo + 0.9 * span
    starts = np.where((x[:-1] < t10) & (x[1:] >= t10))[0]
    times = []
    window = int(0.45 * fs)
    for i in starts[2:52]:                 # 跳过开头两个沿：可能起始就在半途
        seg = x[i:i + window]
        hit = np.where(seg >= t90)[0]
        if hit.size:
            times.append(hit[0] / fs * 1000.0)
    return float(np.median(times)) if times else 0.0


def extract_features(samples: np.ndarray, sample_rate: int = 10000) -> dict[str, float]:
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 16 or sample_rate <= 0:
        return {k: 0.0 for k in FEATURES}

    xc = x - x.mean()                       # 去直流
    n = xc.size
    dur = n / sample_rate

    # ── 幅度 ──
    vpp = float(np.percentile(x, 99) - np.percentile(x, 1))   # 摆幅，不截断
    rms = float(np.sqrt(np.mean(xc**2)))
    peak = float(np.max(np.abs(xc)))
    crest = peak / rms if rms > 1e-12 else 0.0

    # ── 占空比：用 1%/99% 分位当上下轨，避免单个尖峰影响 ──
    lo, hi = np.percentile(x, [1, 99])
    span = float(hi - lo)
    duty = float(np.mean(x > (hi + lo) / 2.0)) if span > 1e-6 else 0.0

    # ── 频域 ──
    f0 = _fundamental(xc, sample_rate)
    mags = np.abs(np.fft.rfft(xc))
    power = mags**2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total = float(power.sum())

    # 谱平坦度：几何均值 / 算术均值。白噪声≈0.56，单频→0
    if total <= 1e-20:
        flatness = 0.0
    else:
        p = power[1:] + 1e-20
        flatness = float(np.exp(np.mean(np.log(p))) / np.mean(p))

    # 相对谱宽：只在"显著谱线"上算，否则噪声底噪乘上 (f-f0)^2 会把结果拉爆
    if total <= 1e-20 or f0 <= 0.2:
        spread_rel = 0.0
    else:
        sig = power > 1e-3 * power.max()
        w = power[sig]
        spread = float(np.sqrt(np.sum(w * (freqs[sig] - f0) ** 2) / w.sum()))
        spread_rel = spread / f0

    # ── 时域：过零率与抖动 ──
    crossings = int(np.sum(xc[:-1] * xc[1:] < 0.0))
    zcr = crossings / dur if dur > 0 else 0.0
    zcr_per_cycle = zcr / (2.0 * f0) if f0 > 0.5 else 0.0

    # 抖动：用"上升过零点"代替"峰"，方波的平台和噪声毛刺都不会干扰它
    jitter_rel = 0.0
    if f0 > 0.2:
        rise = np.where((xc[:-1] < 0) & (xc[1:] >= 0))[0]
        if rise.size >= 20:        # 少于 20 个周期，统计量本身就是噪声
            iv = np.diff(rise).astype(np.float64)
            jitter_rel = float(np.std(iv) / np.mean(iv)) if np.mean(iv) > 0 else 0.0

    if rms < SILENT_RMS:                 # 没信号时，比值类特征不再有意义
        crest, duty, spread_rel, zcr_per_cycle, jitter_rel = 0.0, 0.0, 0.0, 0.0, 0.0

    # 量化到各自的"有效分辨率"。
    # 原因：干净方波的 flatness 数值上只有 1e-4 量级，1e-4 与 3e-4 的差别全是
    # FFT 数值噪声；不做这一步，树会拿这种差别当真实依据去分裂（第 3 步实测踩到）。
    return {
        "vmean": round(float(x.mean()), 3),
        "vpp_v": round(vpp, 3),
        "rms_v": round(rms, 3),
        "crest": round(float(crest), 2),
        "duty": round(float(duty), 3),
        "freq_hz": round(float(f0), 3),
        "rise_ms": round(_rise_ms(x, sample_rate), 2),
        "flatness": round(max(float(flatness), 1e-3), 4),
        "spread_rel": round(float(spread_rel), 2),
        "zcr_per_cycle": round(float(zcr_per_cycle), 2),
        "jitter_rel": round(float(jitter_rel), 3),
    }


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(k, 0.0)) for k in FEATURES]
