"""
一键演示：把 13 个工况全部跑一遍，挑三个展开
============================================

用法：python demo.py
输出：终端一张总表 + demo.md（含三个完整案例的波形与解释）
"""

import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LAB, "step2-features"))
sys.path.insert(0, os.path.join(LAB, "step3-models"))
sys.path.insert(0, HERE)

import simulate  # noqa: E402
from features_v2 import FEATURES, extract_features, features_to_vector  # noqa: E402
from cli import ADVICE, ascii_scope, explain  # noqa: E402

MODEL_DIR = os.path.join(LAB, "models")
META = json.load(open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8"))
MODELS = {m: joblib.load(os.path.join(MODEL_DIR, f"{m}.joblib")) for m in ("mode_A", "mode_B", "mode_C")}

rng = np.random.default_rng(11)
rows = []
for mode, classes in simulate.CLASSES.items():
    model, meta = MODELS[mode], META[mode]
    gen = simulate.GENERATORS[mode]
    for cls in classes:
        s = simulate.new_session(rng)
        x = simulate._finish(gen(cls, s, rng), s, rng)
        feat = extract_features(x, simulate.FS)
        vec = np.array(features_to_vector(feat)).reshape(1, -1)
        proba = model.predict_proba(vec)[0]
        order = np.argsort(proba)[::-1]
        pred = str(model.classes_[order[0]])
        conf = float(proba[order[0]])
        rows.append({"mode": mode, "cls": cls, "pred": pred, "conf": conf,
                     "x": x, "feat": feat, "meta": meta,
                     "top3": [(str(model.classes_[i]), float(proba[i])) for i in order[:3]]})

ok = sum(r["pred"] == r["cls"] for r in rows)
print(f"demo: {ok}/{len(rows)} correct\n")
print(f"{'mode':<8}{'case':<16}{'predicted':<16}{'conf':>6}  ok")
for r in rows:
    print(f"{r['mode']:<8}{r['cls']:<16}{r['pred']:<16}{r['conf']:>6.0%}  {'Y' if r['pred'] == r['cls'] else 'N'}")

# ── 写 demo.md ──
L = ["# 改进版演示\n",
     f"- 13 个工况逐一判定：**{ok}/{len(rows)} 正确**",
     "- 模型：每个模式一个 RandomForest（11 个特征）；模式由操作者指定\n",
     "## 一、总表\n",
     "| 模式 | 工况 | 判定 | 置信度 | 正确 |",
     "|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['mode']} | {r['cls']} | {r['pred']} | {r['conf']:.0%} | "
             f"{'✅' if r['pred'] == r['cls'] else '❌'} |")

SHOW = [("mode_A", "nominal"), ("mode_B", "cap_missing"), ("mode_C", "chatter")]
L.append("\n## 二、三个展开案例\n")
for mode, cls in SHOW:
    r = next(x for x in rows if x["mode"] == mode and x["cls"] == cls)
    L.append(f"\n### {mode} / {cls} → {r['pred']}（{r['conf']:.0%}）\n")
    L.append("```")
    L.append(ascii_scope(r["x"], simulate.FS))
    L.append("```")
    ref = r["meta"]["nominal_range"]
    L.append("| 特征 | 实测 | 正常区间 |")
    L.append("|---|---|---|")
    for k in FEATURES:
        lo, hi = ref[k]
        L.append(f"| {k} | {r['feat'][k]:.4g} | {lo:.4g} – {hi:.4g} |")
    L.append("\n" + explain(mode[-1], r["pred"], r["feat"], r["meta"], r["conf"]) + "\n")

fails = [r for r in rows if r["pred"] != r["cls"]]
if fails:
    L.append("")
    L.append("## 三、没判对的那一个")
    L.append("")
    for r in fails:
        f = r["feat"]
        L.append(f"- {r['mode']} / {r['cls']} → 判成 **{r['pred']}**（{r['conf']:.0%}）；"
                 f"实测 freq={f['freq_hz']:.2f} Hz、duty={f['duty']:.3f}、crest={f['crest']:.2f}")
    L.append("")
    L.append("这不是程序错误，而是两类在某个特征上本来就贴着："
             "频率说它是故障，占空比说它更像正常，模型信了后者。"
             "修法有两条——让 555 模型以频率为主判据，或把 R2 的标称值也作为已知条件喂进去。")
    L.append("")

with open(os.path.join(HERE, "demo.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print("\nwrote demo.md")
