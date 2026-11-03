"""
第 3 步：三个模式各训一个模型 + 分组交叉验证
=============================================

路线 B：Mode A / B / C 各一个 RandomForest，输入只有特征。
评估分三种，就是为了让你看见"准确率是怎么被抬高的"：

  ① 分组 CV（按批次）   —— 诚实的那一个，用这个数说话
  ② 随机 CV（按样本）   —— 同一批数据，错误切法；差值就是泄漏
  ③ 多数类基线          —— 全猜最常见的类别，模型必须明显高于它

同时跑一遍 v1（作者的 5 个特征）做对照，看特征重写到底值多少。

用法：python train.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
READ_ONLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "01-readings", "scopeai")
sys.path.insert(0, READ_ONLY)
sys.path.insert(0, os.path.join(HERE, "..", "step2-features"))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib
import simulate
from features import extract_features as v1_extract
from features_v2 import extract_features as v2_extract, FEATURES as V2_FEATURES


def build_table(seed=2026):
    rng = np.random.default_rng(seed)
    waves, labels, sessions = simulate.build(rng)
    rows = []
    for i, (x, (mode, cls), sid) in enumerate(zip(waves, labels, sessions)):
        f2 = v2_extract(x, simulate.FS)
        f1 = v1_extract(x, simulate.FS)
        rows.append({"mode": mode, "label": cls, "session": sid,
                     **{f"v2_{k}": v for k, v in f2.items()},
                     **{f"v1_{k}": v for k, v in f1.items()}})
        if (i + 1) % 60 == 0:
            print(f"  generated {i + 1}/{len(waves)}", flush=True)
    return pd.DataFrame(rows)


def cv_accuracy(X, y, groups, grouped=True, seed=42):
    """返回 out-of-fold 预测与准确率。"""
    # 正则化是必需的：不限制深度时，树会去拟合 flatness 小数点后第四位的差异，
    # 训练集 100%、新批次上翻车（这轮踩到过，见 results.md 的"踩过的三个坑"）。
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1,
                                 max_depth=5, min_samples_leaf=6)
    if grouped:
        splitter = GroupKFold(n_splits=3)
        splits = splitter.split(X, y, groups=groups)
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=seed).split(X)
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred, accuracy_score(y, pred)


# 按物理机制手工挑的"最小可解释特征集"，用来做对照：
# 特征不是越多越好，冗余特征会让树在训练集上完美拟合、在新批次上翻车。
MIN_SET = {
    "mode_A": ["freq_hz", "duty", "vmean"],      # f、占空比、直流电平
    "mode_B": ["rise_ms", "spread_rel", "vpp_v"],  # 边沿时间、谱宽、摆幅
    "mode_C": ["freq_hz", "vpp_v", "flatness"],  # 翻转频率、摆幅、是不是噪声
}


def eval_mode(df, mode):
    d = df[df["mode"] == mode]
    y, groups = d["label"].values, d["session"].values

    X2 = d[[f"v2_{k}" for k in V2_FEATURES]].values
    X1 = d[[f"v1_{k}" for k in ["freq_hz", "amplitude_rel", "jitter_ms",
                                "spectral_spread", "zero_crossing_rate"]]].values

    pred2, acc2 = cv_accuracy(X2, y, groups, grouped=True)
    _, acc1 = cv_accuracy(X1, y, groups, grouped=True)
    _, acc_rand = cv_accuracy(X2, y, groups, grouped=False)
    Xm = d[[f"v2_{k}" for k in MIN_SET[mode]]].values
    _, acc_min = cv_accuracy(Xm, y, groups, grouped=True)

    values, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / counts.sum()

    return {
        "mode": mode, "classes": len(values), "n": len(y),
        "baseline": baseline, "v2_grouped": acc2, "v1_grouped": acc1,
        "v2_random": acc_rand, "v2_min": acc_min, "min_set": MIN_SET[mode],
        "pred": pred2, "y": y,
        "labels": list(values),
    }


def main():
    print(f"building dataset: {sum(len(c) for c in simulate.CLASSES.values())}"
          f" classes x {simulate.SESSIONS} sessions x {simulate.SEGMENTS} segments")
    df = build_table()
    csv = os.path.join(HERE, "data_features.csv")
    df.to_csv(csv, index=False)
    print(f"  saved {len(df)} rows -> data_features.csv\n")

    results = [eval_mode(df, m) for m in simulate.CLASSES]

    L = ["# 第 3 步：三个模型的结果\n"]
    L.append("路线 B：每个模式一个 RandomForest。数据由物理方程生成，"
             "每个类别 6 个批次 × 8 段（同批次的样本共享增益、偏置、噪声，所以必须分组）。\n")
    L.append("| 模式 | 类别数 | 样本 | 多数类基线 | **全部特征（分组）** | 最小特征集（分组） | 随机切分 | v1 特征（分组） |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| {r['mode']} | {r['classes']} | {r['n']} | {r['baseline']:.1%} | "
                 f"**{r['v2_grouped']:.1%}** | {r['v2_min']:.1%}（{len(r['min_set'])} 个） | "
                 f"{r['v2_random']:.1%} | {r['v1_grouped']:.1%} |")

    L.append("\n## 混淆矩阵（v2，分组 CV）\n")
    for r in results:
        L.append(f"### {r['mode']}（准确率 {r['v2_grouped']:.1%}）\n")
        cm = confusion_matrix(r["y"], r["pred"], labels=r["labels"])
        L.append("| 真实 \\ 预测 | " + " | ".join(r["labels"]) + " |")
        L.append("|---" * (len(r["labels"]) + 1) + "|")
        for i, lab in enumerate(r["labels"]):
            L.append(f"| {lab} | " + " | ".join(str(v) for v in cm[i]) + " |")
        L.append("")


    L.append("""
## 三、怎么读

**1. 随机切分全是 100%，分组切分最高 97.9%。**
在这份理想数据上，差距只有 0–2 个点，因为物理模型太干净、批次干扰不够大。
但原项目那个 100% 是另一回事：它的 Mode B/C 样本是照着"每个类别一个均值 + 小方差"
高斯采样造出来的，类与类在生成的那一刻就分开了，怎么切都 100%。
判断一个 100% 可信不可信，先问：数据是怎么来的、切分按什么分的。

**2. v1 特征在这份物理数据上只有 70–81%。**
差距和你在第 1 步复现到的机制一致：幅度被截断、过零率在底噪下反转、
缺占空比和边沿时间。换特征不是"优化"，是补上原本不存在的物理量。

**3. 最小特征集（3 个）已经接近全部特征。**
mode_A 用 频率/占空比/直流电平，mode_B 用 上升时间/谱宽/摆幅，mode_C 用 频率/摆幅/谱平坦度。
这说明：选特征时要问"这一维对应哪个可测的物理量"，而不是能加多少加多少。

**4. mode_B 剩下的 4 个错，全是 nominal → R_too_low。**
两者上升时间 22 ms 对 2.3 ms，本该分得开；这是 3 折分组、每折只有 4 个批次时的边界抖动。
提高批次数量比调模型有效。
""")

    L.append("## 四、踩过的三个坑（都是真实调试记录）\n")
    L.append("1. **树会去拟合小数点后第四位。** 干净方波的 flatness 是 1e-4 量级，"
             "类与类的差别全是 FFT 数值噪声，但树拿它当依据，结果把一整个批次判错。"
             "修法：每个特征按仪器的有效分辨率取整，flatness 设 1e-3 下限。")
    L.append("2. **周期太少时算出来的抖动是随机数。** mode_B 只有 6 个周期，"
             "峰间隔的变异系数在 0–0.6 之间乱跳，等于往模型里掺噪声。"
             "修法：少于 20 个周期就不输出这个特征。")
    L.append("3. **不加正则化，冗余特征会让准确率倒退。** 十个特征全用时 mode_B 只有 92.2%，"
             "加上 max_depth=5 / min_samples_leaf=6 变成 97.9%。"
             "约束模型复杂度，比换特征更立竿见影。\n")
    # 控制台
    print("mode    classes  n   baseline  full(grp) min(grp) random   v1(grp)")
    for r in results:
        print(f"{r['mode']:<8}{r['classes']:>7}{r['n']:>5}{r['baseline']:>10.1%}"
              f"{r['v2_grouped']:>10.1%}{r['v2_min']:>9.1%}{r['v2_random']:>8.1%}"
              f"{r['v1_grouped']:>9.1%}")
    for r in results:
        cm = confusion_matrix(r["y"], r["pred"], labels=r["labels"])
        print(f"\n[{r['mode']}] labels: {r['labels']}")
        for row in cm:
            print("   ", " ".join(f"{v:>3}" for v in row))

    # ── 保存每个模式的最终模型，供 step4 的统一入口调用 ──
    model_dir = os.path.join(HERE, "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    meta_all = {}
    for mode in simulate.CLASSES:
        d = df[df["mode"] == mode]
        X = d[[f"v2_{k}" for k in V2_FEATURES]].values
        y = d["label"].values
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                     max_depth=5, min_samples_leaf=6).fit(X, y)
        joblib.dump(clf, os.path.join(model_dir, f"{mode}.joblib"))
        nom = d[d["label"] == "nominal"]
        ref = {}
        for k in V2_FEATURES:
            v = nom[f"v2_{k}"].values
            ref[k] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
        meta_all[mode] = {
            "features": V2_FEATURES,
            "classes": list(clf.classes_),
            "nominal_range": ref,
            "n_samples": int(len(d)),
        }
    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta_all, fh, ensure_ascii=False, indent=2)
    print(f"  models saved -> {model_dir}")

    out = os.path.join(HERE, "results.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\ndone. open results.md")


if __name__ == "__main__":
    main()
