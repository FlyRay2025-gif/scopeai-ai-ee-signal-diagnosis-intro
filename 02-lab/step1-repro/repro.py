"""
第 1 步：复现 ScopeAI 的三个错判
================================

做的事情和 app.py 完全一样，只是把中间结果全打出来，方便你逐行看数：

    capture(仿真波形) -> extract_features(5 个特征) -> RandomForest(13 类) -> 概率

只看，不改。脚本跑完会生成 report.md（中文完整报告），终端只打印简短进度，
这样不管终端是什么编码都不会乱码。

运行：  python repro.py
"""

import os
import sys
import warnings

os.environ["SCOPEAI_SIMULATE"] = "1"          # 强制仿真，别去试着连 AD2
warnings.filterwarnings("ignore")             # 屏蔽 sklearn 版本不一致的警告

READ_ONLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "01-readings", "scopeai")   # 只读副本
sys.path.insert(0, READ_ONLY)

import joblib
import numpy as np

import capture
from features import extract_features, features_to_vector

np.random.seed(2026)   # 固定噪声，保证每次跑出来的数一样，方便和参考答案逐位对照

DURATION = 6.0                                # 与作者训练时一致，别改
FAULTS = ["nominal", "R_too_high", "R_too_low", "cap_missing", "no_oscillation", "chatter"]
FS = capture.get_device_info()["sample_rate"]

clf = joblib.load(os.path.join(READ_ONLY, "models", "fault_classifier.pkl"))
le = joblib.load(os.path.join(READ_ONLY, "models", "label_encoder.pkl"))


def ascii_scope(x, fs, window=1.0, width=72, height=10, vmax=2.0):
    """字符示波器：1 秒窗口、纵轴固定 0~2V。拿掉按自身最大值缩放的毛病，
    不然"只有噪声"看起来也像满屏波形。"""
    x = np.clip(x[:int(fs * window)], 0.0, vmax)
    grid = [[" "] * width for _ in range(height)]
    for i, col in enumerate(np.array_split(x, width)):
        top = int(round((vmax - float(np.max(col))) / vmax * (height - 1)))
        bot = int(round((vmax - float(np.min(col))) / vmax * (height - 1)))
        for r in range(top, bot + 1):
            grid[r][i] = "#" if r in (top, bot) else "."
    return "\n".join("".join(row) for row in grid)


rows = []
for fault in FAULTS:
    capture.set_simulated_fault(fault)
    print(f"[{fault}] capturing {DURATION:.0f}s ...", flush=True)
    samples, fs = capture.capture_snapshot(DURATION)

    feat = extract_features(samples, sample_rate=fs)
    vec = np.array(features_to_vector(feat)).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    pred = str(le.inverse_transform([clf.predict(vec)[0]])[0])
    top3 = [(str(le.inverse_transform([i])[0]), proba[i])
            for i in np.argsort(proba)[::-1][:3]]

    print(f"  -> predicted {pred} ({proba.max():.0%})")
    rows.append({"fault": fault, "pred": pred, "conf": proba.max(),
                 "top3": top3, "feat": feat, "wave": ascii_scope(samples, fs)})

# 附加验证：amplitude_rel = ptp/2 再 clip(0,1)，摆幅一大就饱和
saturation = []
t = np.arange(int(2 * 1e4)) / 1e4
for swing in (1.0, 2.0, 4.0, 8.0):
    x = np.where(np.mod(t * 2.5, 1.0) < 0.6, swing, 0.0)
    saturation.append((swing, extract_features(x, sample_rate=10000)["amplitude_rel"]))

# ── 生成中文报告 ────────────────────────────────────────────────────────────
L = []
L.append("# 第 1 步复现结果\n")
L.append(f"- 设备：{capture.get_device_info()['device_name']}（仿真）")
L.append(f"- 采样率：{FS} Hz，单次采集 {DURATION:.0f} s")
L.append(f"- 模型：models/fault_classifier.pkl（作者预训练，sklearn 1.8.0）\n")

L.append("## 一、逐故障复现\n")
for r in rows:
    f = r["feat"]
    # 脚本不给"对/错"的结论：判定对错由你自己写。软件只负责把证据摆出来。
    L.append(f"### {r['fault']} → {r['pred']}（置信度 {r['conf']:.0%}）\n")
    L.append("```")
    L.append(r["wave"])
    L.append("```")
    L.append(f"- 前三名：" + "、".join(f"{n} {p:.0%}" for n, p in r["top3"]))
    L.append(f"- 特征：freq={f['freq_hz']:.2f} Hz，amp={f['amplitude_rel']:.4f}，"
             f"jitter={f['jitter_ms']:.2f} ms，spread={f['spectral_spread']:.1f}，"
             f"zcr={f['zero_crossing_rate']:.1f}/s\n")

L.append("## 二、六种情况的特征总表\n")
L.append("同一列数值几乎不变的，这个特征就没有区分力。最后一列空着，你自己填。\n")
L.append("| 仿真设定 | freq Hz | amp | jitter ms | spread | zcr /s | 判定 | 你的判断 |")
L.append("|---|---|---|---|---|---|---|---|")
for r in rows:
    f = r["feat"]
    L.append(f"| {r['fault']} | {f['freq_hz']:.2f} | {f['amplitude_rel']:.4f} | "
             f"{f['jitter_ms']:.2f} | {f['spectral_spread']:.1f} | "
             f"{f['zero_crossing_rate']:.1f} | {r['pred']} |  |")
L.append(f"\n参考：采样率 {FS} Hz，纯噪声的过零率上限约 {FS/2:.0f}/s。")
L.append("参考：训练时各类别的过零率见 `01-readings/scopeai/scopeai_pipeline.py`"
         " 第 143–171 行。\n")

L.append("## 三、幅度特征饱和验证\n")
L.append("| 方波摆幅 V | amplitude_rel |")
L.append("|---|---|")
for swing, amp in saturation:
    L.append(f"| {swing:.1f} | {amp:.4f} |")
L.append("\n（ptp/2 之后 clip 到 1.0，摆幅超过 2V 就全部记成 1.0）\n")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.md")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))

print("\ndone. open report.md in this folder.")
