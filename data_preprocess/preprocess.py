#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE BigData 2026 - Explainable Suicide Risk Detection
数据预处理 + 固定 GroupKFold 划分

用法（Mac terminal）:
    python3 preprocess.py                 # train.xlsx 与脚本同目录
    python3 preprocess.py /path/train.xlsx
产物:
    train_clean.csv   清洗后的数据，含 fold 列(0~4) 与 24 个因子的 0/1 列
"""

import sys, subprocess, importlib

# ---------- 0. 依赖自检，缺了自动装 ----------
def ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[安装依赖] 正在安装 {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p, n in [("pandas", "pandas"), ("scikit-learn", "sklearn"), ("openpyxl", "openpyxl")]:
    ensure(p, n)

import ast
from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---------- 1. 配置 ----------
INPUT = sys.argv[1] if len(sys.argv) > 1 else "train.xlsx"
OUTPUT = "train_clean.csv"
N_FOLDS = 5
SEED = 42

# 官方 24 类因子（顺序固定，后续所有代码以此为准）
FACTORS = [
    "mental health issues", "physical health/characteristic", "substance use",
    "hopelessness", "emotion dysregulation", "low self-esteem",
    "poor school performance", "low socio-economic status", "interpersonal violence",
    "prior self-harm or suicidal thought/attempt", "poor social support",
    "interpersonal difficulty", "dysfunctional family", "exposure to others' suicide",
    "stressful life event", "traumatic experience", "cognitive deficits",
    "suicide means (with access)", "sexual orientation related issues",
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
]
RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]

print("=" * 60)
print("读取:", INPUT)
df = pd.read_excel(INPUT)
print(f"原始: {len(df)} 行, {df.shape[1]} 列")
print("列名:", list(df.columns))

# ---------- 2. 风险标签规范化（修大小写 + 空格分裂）----------
risk_col = "suicide risk"
df["risk_level"] = df[risk_col].astype(str).str.strip().str.lower()
bad = set(df["risk_level"].unique()) - set(RISK_LEVELS)
assert not bad, f"出现未知风险标签: {bad}"
print("\n[风险标签] 规范化后分布:")
for lv in RISK_LEVELS:
    n = (df["risk_level"] == lv).sum()
    print(f"   {lv:10s} {n:5d}  ({n/len(df)*100:4.1f}%)")

# ---------- 3. factors 解析 + 去重 -> 24 列 0/1 ----------
def parse_factors(v):
    if not isinstance(v, str):
        return []
    try:
        lst = ast.literal_eval(v)
    except Exception:
        return []
    return sorted(set(lst))  # set 去重

df["factors_list"] = df["factors"].apply(parse_factors)

unknown = Counter()
for lst in df["factors_list"]:
    for f in lst:
        if f not in FACTORS:
            unknown[f] += 1
assert not unknown, f"出现未知因子标签: {dict(unknown)}"

for f in FACTORS:
    col = "f_" + f
    df[col] = df["factors_list"].apply(lambda lst, ff=f: int(ff in lst))

n_factor_cols = [("f_" + f) for f in FACTORS]
df["n_factors"] = df[n_factor_cols].sum(axis=1)
print(f"\n[因子] 解析完成，24 列 0/1 已生成；每帖因子数: "
      f"min={df['n_factors'].min()}, max={df['n_factors'].max()}, "
      f"mean={df['n_factors'].mean():.2f}, 0个因子的帖数={int((df['n_factors']==0).sum())}")

# ---------- 4. evidence 缺失标记 ----------
ev = df["evidence for suicide risk level"]
df["evidence"] = ev.astype(str).str.strip()
df["evidence_missing"] = ev.isna() | (df["evidence"].isin(["", "nan"]))
print(f"\n[证据] evidence 缺失行数: {int(df['evidence_missing'].sum())} （标记不删除）")

# ---------- 5. StratifiedGroupKFold: 按 user 分组 + 按 risk_level 分层 ----------
user_col = "anon_user_id"
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(
        sgkf.split(df, y=df["risk_level"], groups=df[user_col])):
    df.iloc[val_idx, df.columns.get_loc("fold")] = fold

assert (df["fold"] >= 0).all(), "存在未分配 fold 的行"
# 验证：同一用户不跨 fold（分组约束必须仍然成立）
leak = df.groupby(user_col)["fold"].nunique()
assert (leak == 1).all(), "数据泄漏：有用户横跨多个 fold！"
print(f"\n[Fold] StratifiedGroupKFold 完成：按 {user_col} 分组 + 按 risk_level 分层，"
      f"{N_FOLDS} 折，无用户跨 fold ✓")

# ---------- 6. 体检报告：各 fold 分布 ----------
print("\n[体检] 各 fold 的规模与风险分布:")
print(f"{'fold':>4} {'帖数':>6} {'用户':>5} | " +
      " ".join(f"{lv[:3]:>6}" for lv in RISK_LEVELS))
for fold in range(N_FOLDS):
    sub = df[df["fold"] == fold]
    dist = [f"{(sub['risk_level']==lv).mean()*100:5.1f}%" for lv in RISK_LEVELS]
    print(f"{fold:>4} {len(sub):>6} {sub[user_col].nunique():>5} | " + " ".join(dist))

print("\n[体检] 稀有因子在各 fold 的正样本数（确认没有 fold 缺类）:")
rare = ["sexual orientation related issues", "exposure to others' suicide",
        "poor school performance", "substance use", "cognitive deficits"]
print(f"{'factor':<42} " + " ".join(f"f{fold}" for fold in range(N_FOLDS)))
for f in rare:
    col = "f_" + f
    counts = [int(df[(df['fold']==fold)][col].sum()) for fold in range(N_FOLDS)]
    print(f"{f:<42} " + " ".join(f"{c:>2}" for c in counts))

# ---------- 7. 保存 ----------
keep = ["row_id", "anon_user_id", "post_id", "post",
        "risk_level", "evidence", "evidence_missing", "fold"] + n_factor_cols
out = df[keep].copy()
out.to_csv(OUTPUT, index=False)
print("\n" + "=" * 60)
print(f"已保存: {OUTPUT}  ({len(out)} 行, {out.shape[1]} 列)")
print("列结构: row_id, anon_user_id, post_id, post, risk_level,")
print("        evidence, evidence_missing, fold, 以及 24 个 f_* 因子列")
print("=" * 60)
