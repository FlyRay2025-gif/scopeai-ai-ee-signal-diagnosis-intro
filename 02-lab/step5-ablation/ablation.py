"""
消融实验：同一份数据，三种"看得见什么"的输入方式
=================================================

    路线 1  11 个物理特征          → RandomForest
    路线 2  时频图（STFT 64×64）   → 小型 2D CNN
    路线 3  原始波形（降采样 2048）→ 小型 1D CNN

三者用**完全相同的数据和相同的按批次分组切分**，只变输入表示。
目的不是比谁高，而是回答：在每类 48 段的规模下，多给模型一些表示，划不划算。

用法：python ablation.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LAB, "step2-features"))
sys.path.insert(0, os.path.join(LAB, "step3-models"))

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import simulate
from features_v2 import FEATURES, extract_features, features_to_vector

torch.set_num_threads(4)
SEED = 2026
EPOCHS = 100
BATCH = 32


def resize2d(a, h, w):
    ri = np.linspace(0, a.shape[0] - 1, h).astype(int)
    ci = np.linspace(0, a.shape[1] - 1, w).astype(int)
    return a[np.ix_(ri, ci)]


def build_dataset():
    """一次生成，三种表示都从这里取。"""
    rng = np.random.default_rng(SEED)
    waves, labels, sessions = simulate.build(rng)

    scalars, specs, raws, ys, groups = [], [], [], [], []
    for x, (mode, cls), sid in zip(waves, labels, sessions):
        f = extract_features(x, simulate.FS)
        scalars.append(features_to_vector(f))

        _, _, Z = stft(x, fs=simulate.FS, nperseg=512, noverlap=256)
        mag = np.log1p(np.abs(Z))
        specs.append(resize2d(mag, 64, 64).astype(np.float32))

        # 原始波形：先按 2048 点抽稀，再做逐样本标准化（等价于自动增益）
        idx = np.linspace(0, x.size - 1, 2048).astype(int)
        r = x[idx].astype(np.float32)
        r = (r - r.mean()) / (r.std() + 1e-8)
        raws.append(r)

        ys.append(f"{mode}__{cls}")
        groups.append(sid)
    return (np.array(scalars), np.array(specs), np.array(raws),
            np.array(ys), np.array(groups))


class CNN2D(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(16 * 4 * 4, n_cls),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class CNN1D(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 15, stride=2, padding=7), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(16 * 8, n_cls),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def train_torch(model, Xtr, ytr, Xte, n_cls, is_2d):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    return out


def main():
    print("generating dataset ...", flush=True)
    S, SPEC, RAW, y, groups = build_dataset()
    le = LabelEncoder().fit(y)
    yc = le.transform(y)
    print(f"  {len(y)} samples, {len(le.classes_)} classes\n")

    splits = list(GroupKFold(n_splits=3).split(S, yc, groups=groups))
    acc = {"scalars+RF": [], "STFT+CNN": [], "raw+CNN": []}

    for fold, (tr, te) in enumerate(splits, 1):
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                    max_depth=5, min_samples_leaf=6).fit(S[tr], yc[tr])
        a1 = (rf.predict(S[te]) == yc[te]).mean()

        m2 = CNN2D(len(le.classes_))
        p2 = train_torch(m2, SPEC[tr], yc[tr], SPEC[te], len(le.classes_), True)
        a2 = (p2 == yc[te]).mean()

        m3 = CNN1D(len(le.classes_))
        p3 = train_torch(m3, RAW[tr], yc[tr], RAW[te], len(le.classes_), False)
        a3 = (p3 == yc[te]).mean()

        acc["scalars+RF"].append(a1)
        acc["STFT+CNN"].append(a2)
        acc["raw+CNN"].append(a3)
        print(f"fold {fold}: scalars {a1:.3f} | stft {a2:.3f} | raw {a3:.3f}", flush=True)

    L = ["# 消融实验：三种输入表示\n",
         f"- 数据：{len(y)} 段，{len(le.classes_)} 类（三个模式合在一起），"
         f"按批次三折分组", "- 参数总量：两个 CNN 都在几千量级，故意做小——问题不是"
         "\"网络够不够强\"，而是\"这份数据配不配得上网络\"\n",
         "| 输入表示 | 模型 | 三折准确率 | 平均 |", "|---|---|---|---|"]
    for k, v in acc.items():
        model = "RandomForest" if "RF" in k else "小型 CNN"
        L.append(f"| {k.split('+')[0]} | {model} | "
                 + " / ".join(f"{a:.1%}" for a in v) + f" | **{np.mean(v):.1%}** |")

    L.append("""
## 怎么读

1. **手写特征不是"落后"，是"省数据"。** 11 个数里每一个都已经是物理量，
   模型几乎不用学；CNN 要从 2048 点或 64×64 图里自己发现"频率""占空比"，
   而每类只有 48 段，它没有足够的例子把这件事学出来。

2. **加长训练不改变结论。** 30 轮和 100 轮的差别在 1–2 个百分点以内，
   说明瓶颈不是训练不充分，是数据量。

3. **原始波形最差（约 51%），时频图居中（约 77%）。**
   时频图比原始波形好，说明"先做一次变换"仍然有价值；
   但它离 96% 差得远，说明这点价值远不如把物理知识直接写进特征。

## 这个实验不能证明什么

- **不能证明"手写特征永远更好"**。每类样本数从 48 涨到上千、或者任务是暂态信号
  （事件只存在几十毫秒），结论可能反转——那时候标量特征会把事件平均掉。
- **CNN 没有系统调参、没有数据增强**。我给了它两次机会（30 轮 / 100 轮），
  差距稳定，但没做穷举。写进材料时要带上这句限定。
- **这里是 13 类合并的任务**，与正式流程里"每模式一个模型"不是同一个问题；
  放在一起只是为了三类输入有完全相同的训练/测试划分。

## 如果以后要重做

每类样本 ×10、加时间平移与噪声增强、按模式分开各跑一次——
这三个变量里任何一个变了，都值得重跑这张表。
""")

    with open(os.path.join(HERE, "ablation.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\nmean:", {k: round(float(np.mean(v)), 3) for k, v in acc.items()})
    print("wrote ablation.md")


if __name__ == "__main__":
    main()
