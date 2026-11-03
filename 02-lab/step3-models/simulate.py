"""
第 3 步：物理波形生成器
========================

和作者那版"照着手填的均值采样"不同，这里每一类都由**电路方程**算出来，
再叠加真实采集会有的干扰：

    元件容差  ±3%     → 每个批次重抽一次
    通道增益  ±6%     → 同上
    直流偏置  ±80 mV  → 同上
    噪声      3–20 mV → 每段不同
    12 位 ADC 量化

"批次"（session）= 一次装配、一次接线、一批连续采集。
同一个批次里的所有样本共享增益/偏置/容差，所以交叉验证必须按批次分组，
否则模型会通过"这批的增益"认人，而不是通过波形认故障。
"""

from __future__ import annotations

import numpy as np

FS = 10000
DUR = 6.0
N = int(FS * DUR)

SESSIONS = 6          # 每个类别 6 个批次
SEGMENTS = 8          # 每个批次 8 段

CLASSES = {
    "mode_A": ["nominal", "R_too_high", "R_too_low", "cap_missing", "no_oscillation"],
    "mode_B": ["nominal", "R_too_high", "R_too_low", "cap_missing"],
    "mode_C": ["nominal", "chatter", "clipping", "no_oscillation"],
}


def new_session(rng: np.random.Generator) -> dict:
    """一个批次的固定干扰。"""
    return {
        "tol": 1.0 + rng.normal(0.0, 0.03),        # 元件容差
        "gain": 1.0 + rng.normal(0.0, 0.06),       # 通道增益
        "offset": rng.normal(0.0, 0.08),           # 直流偏置
        "noise": rng.uniform(0.003, 0.020),        # 噪声有效值
        "th_drift": rng.normal(0.0, 0.08),         # 比较器阈值漂移
        "phase": rng.uniform(0, 2 * np.pi),        # 采样起点相位
    }


def _finish(v: np.ndarray, s: dict, rng: np.random.Generator) -> np.ndarray:
    """叠加通道增益、偏置、噪声和 12 位量化。"""
    v = 5.0 + s["gain"] * (v - 5.0)
    v = v + s["offset"] + rng.normal(0.0, s["noise"], v.size)
    v = np.round(v / 5.0 * 4095) / 4095 * 5.0          # 12 位 ADC
    return np.clip(v, 0.0, 5.0)


# ── Mode A：NE555 无稳态，f = 1.44/((R1+2R2)C)，占空比 =(R1+R2)/(R1+2R2) ──
def mode_a(cls: str, s: dict, rng: np.random.Generator) -> np.ndarray:
    R1 = 1000.0 * s["tol"]
    C = 47e-6 * s["tol"]
    R2 = {"nominal": 5000.0, "R_too_high": 22000.0, "R_too_low": 470.0}.get(cls)
    if cls == "cap_missing":
        return np.full(N, 4.85)        # 定时电容缺失：阈值永远到不了，输出卡在高电平
    if cls == "no_oscillation":
        return np.full(N, 0.04)        # 没供电 / 输出线断：卡在 0 V
    R2 = R2 * s["tol"]
    f = 1.44 / ((R1 + 2 * R2) * C)
    duty = (R1 + R2) / (R1 + 2 * R2)
    t = np.arange(N) / FS
    return np.where(np.mod(t * f + s["phase"] / (2 * np.pi), 1.0) < duty, 4.9, 0.05)


# ── Mode B：RC 充放电，τ = RC，方波激励周期 1 s ──
def mode_b(cls: str, s: dict, rng: np.random.Generator) -> np.ndarray:
    R, C = 10e3 * s["tol"], 1e-6 * s["tol"]
    tau = {
        "nominal": R * C,             # 10 ms
        "R_too_high": 10 * R * C,     # 100 ms，半个周期内来不及充满
        "R_too_low": 0.1 * R * C,     # 1 ms，几乎跟得上
        "cap_missing": 2e-5,          # 无电容：等于导线，边沿接近垂直
    }[cls]
    t = np.arange(N) / FS
    ph = np.mod(t / 1.0 + s["phase"] / (2 * np.pi), 1.0)
    t_on = np.clip(ph * 1.0, 0.0, None)
    t_off = np.clip((ph - 0.5) * 1.0, 0.0, None)   # 防止负指数溢出
    return np.where(ph < 0.5, 5.0 * (1 - np.exp(-t_on / tau)), 5.0 * np.exp(-t_off / tau))


# ── Mode C：LM339 比较器，输入 5 Hz 正弦，与 2.5 V 阈值比较 ──
def mode_c(cls: str, s: dict, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N) / FS
    amp = {"nominal": 2.0, "chatter": 2.0, "clipping": 6.0, "no_oscillation": 0.0}[cls]
    vin = 2.5 + amp * np.sin(2 * np.pi * 5.0 * t + s["phase"])
    vin = np.clip(vin, 0.0, 5.0)                  # 输入被电源轨钳位
    th = 2.5 + s["th_drift"]

    if cls == "chatter":                          # 无迟滞 + 阈值附近噪声 → 来回抖
        out = (vin + rng.normal(0, 0.18, N)) > th
    else:
        out = vin > th

    if cls == "clipping":                         # 输出摆幅受限 + 电平偏移
        return np.where(out, 3.2, 0.9)
    return np.where(out, 4.4, 0.6)


GENERATORS = {"mode_A": mode_a, "mode_B": mode_b, "mode_C": mode_c}


def build(rng: np.random.Generator):
    """返回 (波形列表, 标签列表, 批次列表)。"""
    waves, labels, sessions = [], [], []
    for mode, classes in CLASSES.items():
        gen = GENERATORS[mode]
        for cls in classes:
            for sid in range(SESSIONS):
                s = new_session(rng)
                for _ in range(SEGMENTS):
                    waves.append(_finish(gen(cls, s, rng), s, rng))
                    labels.append((mode, cls))
                    sessions.append(f"{mode}-{cls}-s{sid}")
    return waves, labels, sessions
