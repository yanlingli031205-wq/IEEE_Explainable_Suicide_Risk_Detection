# %% [markdown]
# ## 9. 收尾保存（放在 notebook 最后，每个模型跑完都运行这格）
# 把本模型的 OOF 缓存、混淆矩阵图、OOF 分数，统一归到 Drive 的 results 文件夹，
# 并把分数追加进一张汇总表 —— 三个模型跑完后一眼能看到对比。

# %%
import os, shutil, numpy as np, pandas as pd

# Drive 里专门放结果的文件夹（PROJECT_DIR 下的 results/）
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1) 确认 OOF 缓存在位，并复制一份到 results/（save_oof 已存到 oof_cache/，这里再归整一份）
src_oof = os.path.join(PROJECT_DIR, "oof_cache", f"oof_1a_{MODEL_TAG}.npz")
dst_oof = os.path.join(RESULTS_DIR, f"oof_1a_{MODEL_TAG}.npz")
if os.path.exists(src_oof):
    shutil.copy(src_oof, dst_oof)
    print(f"✓ OOF 缓存已保存: {dst_oof}")
else:
    # 兜底：直接用内存里的 oof_probs 存
    np.savez(dst_oof, probs=oof_probs, row_ids=oof_rowid)
    print(f"✓ OOF 缓存(从内存)已保存: {dst_oof}")

# 2) 混淆矩阵图归位
src_png = os.path.join(PROJECT_DIR, f"cm_{MODEL_TAG}_1a.png")
if os.path.exists(src_png):
    shutil.copy(src_png, os.path.join(RESULTS_DIR, f"cm_{MODEL_TAG}_1a.png"))
    print(f"✓ 混淆矩阵图已保存: results/cm_{MODEL_TAG}_1a.png")

# 3) 分数追加进汇总表 scores_1a.csv
summary_path = os.path.join(RESULTS_DIR, "scores_1a.csv")
row = {
    "model": MODEL_TAG,
    "oof_weighted_f1": round(float(overall_f1), 4),
    "fold_mean": round(float(np.mean(fold_scores)), 4),
    "fold_std": round(float(np.std(fold_scores)), 4),
    "folds": ";".join(f"{s:.4f}" for s in fold_scores),
}
if os.path.exists(summary_path):
    sdf = pd.read_csv(summary_path)
    sdf = sdf[sdf["model"] != MODEL_TAG]  # 同模型重跑则覆盖旧记录
    sdf = pd.concat([sdf, pd.DataFrame([row])], ignore_index=True)
else:
    sdf = pd.DataFrame([row])
sdf = sdf.sort_values("oof_weighted_f1", ascending=False).reset_index(drop=True)
sdf.to_csv(summary_path, index=False)

print(f"\n✓ 分数已追加到: {summary_path}")
print("\n===== 当前 task1a 各模型 OOF Weighted F1 汇总 =====")
print(sdf.to_string(index=False))
print("\n（三个模型都跑完后，用这张表 + 三个 oof_1a_*.npz 进入融合阶段）")
