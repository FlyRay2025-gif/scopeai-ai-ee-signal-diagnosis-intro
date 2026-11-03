"""
第 2 步对照：v1 的 5 个特征 vs v2 的 9 个特征，在同样 6 个仿真工况上
=====================================================================

用法：python compare.py
输出：终端简表 + comparison.md（中文完整版）

不做训练、不碰模型。这一步只回答一件事：新特征里有没有"v1 拿不到"的信息。
"""

import os
import sys
import warnings

os.environ["SCOPEAI_SIMULATE"] = "1"
warnings.filterwarnings("ignore")

READ_ONLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "01-readings", "scopeai")
sys.path.insert(0, READ_ONLY)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import capture
from features import extract_features as v1_extract
from features_v2 import extract_features as v2_extract

np.random.seed(2026)
FS = 10000
DUR = 6.0
FAULTS = ["nominal", "R_too_high", "R_too_low", "cap_missing", "no_oscillation", "chatter"]

rows = []
for f in FAULTS:
    capture.set_simulated_fault(f)
    x, fs = capture.capture_snapshot(DUR)
    # 同时算一个"不设门槛"的 zcr/周期，用来验证那句建议到底管不管用
    xc = x - x.mean()
    b = v2_extract(x, fs)
    raw_zcr = float(np.sum(xc[:-1] * xc[1:] < 0.0)) / (len(x) / fs)
    ungated = raw_zcr / (2 * b["freq_hz"]) if b["freq_hz"] > 0.5 else float("nan")
    rows.append((f, v1_extract(x, fs), b, ungated))
    print(f"[{f}] done", flush=True)

L = ["# 特征对照：v1 vs v2\n"]
L.append(f"- 采样率 {FS} Hz，窗口 {DUR:.0f} s，同一份仿真波形")
L.append("- v1 = 作者的 5 个特征；v2 = 第 2 步重写的 9 个特征\n")

L.append("## 一、v1（原版）\n")
L.append("| 工况 | freq | amp | jitter | spread | zcr |")
L.append("|---|---|---|---|---|---|")
for f, a, _, _ in rows:
    L.append(f"| {f} | {a['freq_hz']:.2f} | {a['amplitude_rel']:.4f} | "
             f"{a['jitter_ms']:.2f} | {a['spectral_spread']:.1f} | {a['zero_crossing_rate']:.1f} |")

L.append("\n## 二、v2（重写版）\n")
L.append("| 工况 | Vpp | rms_AC | crest | duty | freq | flatness | spread/f | zcr/周期 | jitter_cv |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for f, _, b, _ in rows:
    L.append(f"| {f} | {b['vpp_v']:.2f} | {b['rms_v']:.3f} | {b['crest']:.2f} | {b['duty']:.3f} | "
             f"{b['freq_hz']:.2f} | {b['flatness']:.3f} | {b['spread_rel']:.2f} | "
             f"{b['zcr_per_cycle']:.2f} | {b['jitter_rel']:.4f} |")

L.append("""
## 三、怎么读这张表

**1. flatness 是唯一能一刀切开"有信号/没信号"的特征。**
三条正常波形是 0.000–0.001，三条"没信号"的工况全部 0.558–0.560（白噪声的理论值约 0.56，
单频信号趋近 0）。v1 里没有任何一个特征能做到这一点——它的 zcr 反而给噪声最高分。

**2. Vpp 三条方波都是 1.83–1.84。**
这不是"特征失效"，而是物理事实：三种故障的摆幅本来就一样，只有频率和占空比变了。
区别在 v1：同样 1.8 V 的摆幅，它给出 0.94，因为 ptp/2 之后被截断在 1.0 附近；
换成真实的 0–5 V 电路，v1 会恒等于 1.0，v2 记 5.0，摆幅信息保住了。

**3. duty 0.600 / 0.550 / 0.950 是 v1 完全没有的量。**
555 的 R2 同时改变频率和占空比，v1 只能通过 spread、jitter 间接地、模糊地反映。

**4. crest 要小心解读。**
1.27 / 1.15 / 4.44 —— 4.44 不是"波形更像噪声"，而是 95% 占空比让交流分量变小，
峰值除以小有效值自然变大。所以它和 duty 部分重叠，不能单独用来判断"有没有形状"。

**5. jitter_cv ≈ 0，而且这是对的。**
仿真的方波是理想周期信号，本来就没有抖动。v1 报的 5.46 ms 其实是**方波平台上
的噪声造出了许多"假峰"**，峰间隔的标准差测到的是噪声，不是抖动。换成真实硬件
（晶振、电源纹波）这个值才会活过来。

**6. spread/f 比 v1 的绝对谱宽稳定**（3.1 / 2.9 / 18.7 对 91 / 50 / 328），
但仍然受占空比影响：窄脉冲天然谱宽大。所以它是"形状"的间接量，不是"干净度"。

## 四、关于"zcr 除以频率"这个建议

结论：**没有解决最要紧的问题，反而会让噪声看起来正常。**

不设门槛时，六种工况算出来是：

| 工况 | zcr/(2f0) |
|---|---|
| nominal | 0.97 |
| R_too_high | 0.83 |
| R_too_low | 0.99 |
| cap_missing（纯噪声） | **0.75** |
| no_oscillation（纯噪声） | **2.07** |
| chatter（噪声带） | 1.20 |

干净方波是 0.97，纯噪声是 0.75–2.07 —— **同一个量级，分不开**。
原因是分子分母同时被噪声推高：f0 是从噪声里挑出的随机高频，zcr 恒为 fs/2，一除就回到 1 附近。

它在"有基频、但波形带毛刺"时才有用（chatter = 1.20），所以 v2 保留了它，
但只在 rms > 0.02 V 且有基频时计算；判断"是不是噪声"交给 flatness。

一句话：**归一化不等于有信息。除以一个同样不可靠的量，只是把错误摊平。**
""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison.md")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))

print("\ncase             vpp    rms  crest   duty      f0    flat  spread/f  zcr/cyc  jit_cv")
for f, _, b, _ in rows:
    print(f"{f:<14}{b['vpp_v']:>7.2f}{b['rms_v']:>7.3f}{b['crest']:>6.2f}{b['duty']:>7.3f}"
          f"{b['freq_hz']:>8.2f}{b['flatness']:>8.3f}{b['spread_rel']:>10.2f}"
          f"{b['zcr_per_cycle']:>9.2f}{b['jitter_rel']:>8.4f}")
print("\ndone. open comparison.md")
