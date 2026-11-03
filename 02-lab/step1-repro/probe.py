"""
探针：自己造一个方波，喂给作者的模型，看它判成什么
==================================================

用法：
    python probe.py 5          # 5 Hz 方波
    python probe.py 7 0.95     # 7 Hz，占空比 95%

参数：频率 Hz，占空比（默认 0.95，跟 capture.py 里 R_too_low 的仿真一致）

这个脚本**不碰只读副本**，波形在本地生成。你可以用它扫一遍 3~10 Hz，
找出模型在哪一赫兹"翻脸"——那条边界就是 Q4 的答案。
"""

import os
import sys
import warnings

os.environ["SCOPEAI_SIMULATE"] = "1"
warnings.filterwarnings("ignore")

READ_ONLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "01-readings", "scopeai")
sys.path.insert(0, READ_ONLY)

import joblib
import numpy as np

from features import extract_features, features_to_vector

FS = 10000
DURATION = 6.0

clf = joblib.load(os.path.join(READ_ONLY, "models", "fault_classifier.pkl"))
le = joblib.load(os.path.join(READ_ONLY, "models", "label_encoder.pkl"))


def square(freq, duty=0.95, lo=0.1, hi=1.9):
    n = int(DURATION * FS)
    t = np.arange(n) / FS
    wave = np.where(np.mod(t * freq, 1.0) < duty, hi, lo)
    return wave + np.random.normal(0.0, 0.01, n)


def scope(x, width=72, height=10, vmax=2.0, window=1.0):
    x = np.clip(x[:int(FS * window)], 0, vmax)
    grid = [[" "] * width for _ in range(height)]
    for i, col in enumerate(np.array_split(x, width)):
        top = int(round((vmax - float(np.max(col))) / vmax * (height - 1)))
        bot = int(round((vmax - float(np.min(col))) / vmax * (height - 1)))
        for r in range(top, bot + 1):
            grid[r][i] = "#" if r in (top, bot) else "."
    return "\n".join("".join(row) for row in grid)


freq = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
duty = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95

x = square(freq, duty)
feat = extract_features(x, sample_rate=FS)
vec = np.array(features_to_vector(feat)).reshape(1, -1)
proba = clf.predict_proba(vec)[0]
pred = str(le.inverse_transform([clf.predict(vec)[0]])[0])
top3 = [(str(le.inverse_transform([i])[0]), proba[i]) for i in np.argsort(proba)[::-1][:3]]

print(scope(x))
print(f"\ninput : {freq:.2f} Hz, duty {duty:.2f}")
print(f"feat  : freq={feat['freq_hz']:.2f} amp={feat['amplitude_rel']:.4f} "
      f"jitter={feat['jitter_ms']:.2f} spread={feat['spectral_spread']:.1f} "
      f"zcr={feat['zero_crossing_rate']:.1f}")
print(f"pred  : {pred} ({proba.max():.0%})")
print("top3  : " + " | ".join(f"{n} {p:.0%}" for n, p in top3))
