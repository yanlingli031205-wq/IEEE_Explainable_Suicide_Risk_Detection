#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE BigData 2026 - 共享地基 (common.py)
被 task1a / task2 训练脚本共同 import。封装：
  - 数据加载 (读 train_clean.csv)
  - fold 取用 (复用预处理已切好的 fold 列)
  - 评测函数 (Weighted F1 / Macro F1)
  - 模型与 tokenizer 加载
  - OOF 概率缓存的存/读
Colab 用法: 把 train_clean.csv 上传, 与本文件同目录, 其他脚本 `from common import *`
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report

# ---------- 全局常量 ----------
DATA_PATH = "train_clean.csv"
OOF_DIR = "oof_cache"          # OOF 概率缓存目录
os.makedirs(OOF_DIR, exist_ok=True)

RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RISK2ID = {r: i for i, r in enumerate(RISK_LEVELS)}
ID2RISK = {i: r for r, i in RISK2ID.items()}

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
FACTOR_COLS = ["f_" + f for f in FACTORS]
N_FOLDS = 5

# 候选模型的 HuggingFace 名称
MODELS = {
    # 自杀风险检测需全覆盖证据，已移除 BERTweet(128上限会截断10.6%帖子的关键证据)
    # 只保留能吃长文本(512)的模型
    "deproberta":    "rafalposwiata/deproberta-large-v1",
    "deberta":       "microsoft/deberta-v3-large",
    "roberta":       "roberta-large",
}

# 每个模型支持的最大序列长度。512 覆盖 97.8% 的帖子(256 只覆盖 92.2%)
MODEL_MAX_LEN = {
    "deproberta": 512,
    "deberta":    512,
    "roberta":    512,
}

# ---------- 数据加载 ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    assert "fold" in df.columns, "缺少 fold 列，请先跑 preprocess.py"
    df["label_id"] = df["risk_level"].map(RISK2ID)
    assert df["label_id"].notna().all(), "存在无法映射的 risk_level"
    df["label_id"] = df["label_id"].astype(int)
    return df

def get_fold(df, fold):
    """返回 (train_df, valid_df): valid 是该 fold, train 是其余"""
    tr = df[df["fold"] != fold].reset_index(drop=True)
    va = df[df["fold"] == fold].reset_index(drop=True)
    return tr, va

# ---------- 评测 ----------
def weighted_f1(y_true, y_pred):
    """task 1a 主指标"""
    return f1_score(y_true, y_pred, average="weighted")

def macro_f1_multilabel(y_true, y_pred):
    """task 2 主指标; y_*: (N, 24) 0/1 矩阵"""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def report_risk(y_true, y_pred):
    return classification_report(
        y_true, y_pred, target_names=RISK_LEVELS, digits=4, zero_division=0)

# ---------- OOF 缓存 ----------
def save_oof(model_tag, probs, row_ids, task):
    """probs: 1a 为 (N,4); task2 为 (N,24)。按 row_id 对齐存盘"""
    path = os.path.join(OOF_DIR, f"oof_{task}_{model_tag}.npz")
    np.savez(path, probs=probs, row_ids=np.array(row_ids, dtype=object))
    print(f"[OOF] 已保存 {path}  shape={probs.shape}")

def load_oof(model_tag, task):
    path = os.path.join(OOF_DIR, f"oof_{task}_{model_tag}.npz")
    d = np.load(path, allow_pickle=True)
    return d["probs"], d["row_ids"]

# ---------- 模型加载 ----------
def load_model_and_tokenizer(model_tag, num_labels, problem_type=None):
    """
    problem_type=None -> 单标签(1a); "multi_label_classification" -> task2
    返回 (model, tokenizer)
    DeBERTa-v3 用自定义分类头(绕开缺失的 pooler)；其他模型用标准方式。
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = MODELS[model_tag]
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

    if model_tag == "deberta":
        # DeBERTa-v3 的默认分类流程依赖未预训练的 pooler，会导致退化到多数类。
        # 改为自定义：取 [CLS] 表示 -> dropout -> linear，绕开 pooler。
        import torch.nn as nn
        from transformers import AutoModel, AutoConfig

        class DebertaClassifier(nn.Module):
            def __init__(self, model_name, num_labels, problem_type=None):
                super().__init__()
                self.config = AutoConfig.from_pretrained(model_name)
                self.config.num_labels = num_labels
                self.config.problem_type = problem_type
                self.num_labels = num_labels
                self.problem_type = problem_type
                self.backbone = AutoModel.from_pretrained(model_name)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)

            def forward(self, input_ids=None, attention_mask=None,
                        token_type_ids=None, labels=None, **kwargs):
                out = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]   # 取 [CLS] 位置
                cls = cls.to(self.classifier.weight.dtype)   # 对齐精度
                logits = self.classifier(self.dropout(cls))
                loss = None
                if labels is not None:
                    import torch
                    if self.problem_type == "multi_label_classification":
                        loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                    else:
                        loss = nn.CrossEntropyLoss()(logits, labels.long())
                from transformers.modeling_outputs import SequenceClassifierOutput
                return SequenceClassifierOutput(loss=loss, logits=logits)

        model = DebertaClassifier(name, num_labels, problem_type)
        return model, tokenizer

    # 其他模型(deproberta)走标准方式
    kwargs = {"num_labels": num_labels}
    if problem_type:
        kwargs["problem_type"] = problem_type
    model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    return model, tokenizer

# ---------- 自检 ----------
if __name__ == "__main__":
    df = load_data()
    print(f"数据: {len(df)} 行, fold 分布:")
    print(df["fold"].value_counts().sort_index())
    print(f"\n风险标签 id 映射: {RISK2ID}")
    print(f"因子列数: {len(FACTOR_COLS)}")
    for f in range(N_FOLDS):
        tr, va = get_fold(df, f)
        print(f"fold {f}: train={len(tr)} valid={len(va)} "
              f"valid各类={va['risk_level'].value_counts().to_dict()}")
    print("\n[OK] 共享地基自检通过")
