"""
统一入口：给一段波形 + 一个已知模式，输出故障判定
==================================================

用法：

    # 仿真：指定模式与故障
    python cli.py --mode A --fault R_too_high
    python cli.py --mode C --fault chatter

    # 真实数据：从文件读一段采样（单列 .csv / .txt，或 .npy）
    python cli.py --mode B --file ..\\my_record.csv --fs 10000

    # 看看默认的 nominal 长什么样
    python cli.py --mode A

设计上的两个约定（和第 1 步的结论一致）：

1. **模式由你告诉它**。它不再猜「A 还是 C」——那是信息缺失，不是模型能补的。
2. **AI 解释不参与判断**。这里没有调用任何大模型，诊断文字是从"判定 + 特征"
   按规则拼出来的；要接 qwen，把 explain() 换掉即可，判定结果不会变。
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
MODEL_DIR = os.path.join(LAB, "models")
sys.path.insert(0, os.path.join(LAB, "step2-features"))
sys.path.insert(0, os.path.join(LAB, "step3-models"))

from features_v2 import FEATURES, extract_features, features_to_vector  # noqa: E402

ADVICE = {
    ("mode_A", "nominal"): "555 振荡正常，频率与占空比在标称区间。",
    ("mode_A", "R_too_high"): "频率明显偏低。定时电阻 R2 偏大：把电位器往小拧，或换更小的 R2。",
    ("mode_A", "R_too_low"): "频率明显偏高。定时电阻 R2 偏小：把电位器往大拧，或换更大的 R2。",
    ("mode_A", "cap_missing"): "输出卡在高电平、没有振荡。定时电容缺失：检查 C 是否插在 6 脚与地之间。",
    ("mode_A", "no_oscillation"): "输出贴 0 V。先查供电和 555 的 8/4 脚，再查输出线到探头的连接。",
    ("mode_B", "nominal"): "RC 充电曲线正常，时间常数在标称区间。",
    ("mode_B", "R_too_high"): "充电明显变慢（τ 偏大）。电阻偏大或电容偏大：检查 R、C 的实际值。",
    ("mode_B", "R_too_low"): "充电明显变快（τ 偏小）。电阻偏小：注意它与「电容缺失」只差边沿快慢。",
    ("mode_B", "cap_missing"): "边沿几乎垂直，没有充电过程。电容没接上：检查 C 是否真正接入回路。",
    ("mode_C", "nominal"): "比较器翻转干净，占空比与频率正常。",
    ("mode_C", "chatter"): "在阈值附近反复抖动（无迟滞 + 噪声）。加正反馈迟滞，或把阈值移离噪声带。",
    ("mode_C", "clipping"): "输出摆幅明显变小，被限幅。检查供电电压和输出上拉。",
    ("mode_C", "no_oscillation"): "输出恒定不翻转。输入信号没有到达阈值：查信号源与参考电压。",
}


def ascii_scope(x, fs, width=72, height=10, vmax=5.0):
    x = np.clip(np.asarray(x, dtype=float)[: int(fs * 1.0)], 0, vmax)
    grid = [[" "] * width for _ in range(height)]
    for i, col in enumerate(np.array_split(x, width)):
        top = int(round((vmax - float(np.max(col))) / vmax * (height - 1)))
        bot = int(round((vmax - float(np.min(col))) / vmax * (height - 1)))
        for r in range(top, bot + 1):
            grid[r][i] = "#" if r in (top, bot) else "."
    return "\n".join("".join(row) for row in grid)


def load_waveform(args):
    if args.file:
        if args.file.endswith(".npy"):
            x = np.load(args.file)
        else:
            x = np.loadtxt(args.file, delimiter="," if args.file.endswith(".csv") else None)
        x = np.asarray(x, dtype=float).ravel()
        return x, args.fs, f"file:{os.path.basename(args.file)}"

    import simulate
    rng = np.random.default_rng()
    cls = args.fault
    classes = simulate.CLASSES[f"mode_{args.mode}"]
    if cls not in classes:
        sys.exit(f'fault "{cls}" not in mode {args.mode}; choose from {classes}')
    s = simulate.new_session(rng)
    gen = simulate.GENERATORS[f"mode_{args.mode}"]
    x = simulate._finish(gen(cls, s, rng), s, rng)
    return x, simulate.FS, f"simulation:{cls}"


def explain(mode, fault, feat, meta, confidence):
    parts = [ADVICE.get((f"mode_{mode}", fault), "")]
    if confidence < 0.60:
        parts.append("注意：置信度偏低，建议复测一段，或检查探头是否接牢。")
    if fault == "nominal":
        return parts[0]          # 判成正常时不再列"偏离"，否则会自相矛盾
    ref = meta["nominal_range"]
    dev = []
    for k in ("freq_hz", "duty", "vpp_v", "rise_ms"):
        lo, hi = ref[k]
        v = feat[k]
        if hi - lo < 1e-9:
            continue
        if v < lo - 1e-9:
            dev.append(f"{k} 低于正常范围（{v:.3g} < {lo:.3g}）")
        elif v > hi + 1e-9:
            dev.append(f"{k} 高于正常范围（{v:.3g} > {hi:.3g}）")
    if dev:
        parts.append("偏离正常的量：" + "；".join(dev) + "。")
    return " ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "C"], required=True)
    ap.add_argument("--fault", default="nominal")
    ap.add_argument("--file", default=None)
    ap.add_argument("--fs", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(HERE, "latest_report.md"))
    args = ap.parse_args()

    model_path = os.path.join(MODEL_DIR, f"mode_{args.mode}.joblib")
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        sys.exit("model not found -- run: cd ../step3-models && python train.py")

    clf = joblib.load(model_path)
    meta = json.load(open(meta_path, encoding="utf-8"))[f"mode_{args.mode}"]

    x, fs, source = load_waveform(args)
    feat = extract_features(x, fs)
    vec = np.array(features_to_vector(feat)).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    order = np.argsort(proba)[::-1]
    fault = str(clf.classes_[order[0]])
    conf = float(proba[order[0]])
    top3 = [(str(clf.classes_[i]), float(proba[i])) for i in order[:3]]

    print(f"ScopeAI-lite | mode={args.mode} | source={source}")
    print(ascii_scope(x, fs))
    print("features:", "  ".join(f"{k}={feat[k]:.4g}" for k in FEATURES))
    print("top3    :", " | ".join(f"{n} {p:.0%}" for n, p in top3))
    print(f"result  : {fault} ({conf:.0%})")

    text = explain(args.mode, fault, feat, meta, conf)

    ref = meta["nominal_range"]
    L = [f"# 诊断报告：{source}\n",
         f"- 模式：{args.mode}（由操作者指定，不由模型猜）",
         f"- 采样率：{fs} Hz，样本数：{len(x)}",
         f"- 判定：**{fault}**，置信度 {conf:.0%}",
         f"- 前三名：" + "、".join(f"{n} {p:.0%}" for n, p in top3) + "\n",
         "```", ascii_scope(x, fs), "```\n",
         "## 特征与正常范围\n",
         "| 特征 | 实测 | 正常区间(5%–95%) |", "|---|---|---|"]
    for k in FEATURES:
        lo, hi = ref[k]
        L.append(f"| {k} | {feat[k]:.4g} | {lo:.4g} – {hi:.4g} |")
    L += ["\n## 结论\n", text + "\n",
          "> 判定来自本地的 RandomForest，解释由特征与判定按规则生成；"
          "两者都不调用外部大模型。\n"]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("report  :", os.path.basename(args.out))


if __name__ == "__main__":
    main()
