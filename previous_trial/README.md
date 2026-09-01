# IEEE BigData 2026 — Explainable Suicide Risk Detection

基于 Reddit r/SuicideWatch 帖子、按 C-SSRS 临床量表标注的**可解释自杀风险检测**竞赛方案。当前实时榜 **第 6 名（共 11 队）**，已进入前 8 名（IEEE BigData 论文邀请门槛）。

> 高度敏感领域：本项目用于学术研究与风险识别，方法可靠性优先于单纯刷分。

---

## 任务概览

竞赛包含三个子任务，综合分 = `0.4·风险 + 0.3·证据 + 0.3·因子`：

| 子任务 | 内容 | 指标 | 权重 |
|---|---|---|---|
| **1a** | 风险四分类（Indicator / Ideation / Behavior / Attempt，程度递进的有序分类） | Weighted F1 | 40% |
| **1b** | 证据短语 span 抽取（verbatim，长度 ≤ gold 的 3 倍） | Phrase F1 | 30% |
| **2** | 24 类因子多标签识别 | Macro F1 | 30% |

- 截止：2026-08-31（代码 + 报告），评分 = 性能 30% + 创新 40% + 报告 30%。
- 实时榜每天最多 3 次提交，**最终以隐藏测试集评测** → 以 OOF 为可靠代理选配置，不盲刷 public。

---

## 当前方案与结果

| 任务 | 模型方案 | OOF 分数 |
|---|---|---|
| 1a 风险分类 | **DepRoBERTa + RoBERTa 融合**（概率层加权 0.55 / 0.45） | Weighted F1 **0.8075** |
| 1b 证据抽取 | **DepRoBERTa** BIO token 分类 + indicator 空证据规则 | Phrase F1 **≈ 0.82** |
| 2 因子识别 | **DepRoBERTa** 24 头 sigmoid 多标签 + 逐类阈值校准 | Macro F1 **0.4585** |

**Test 实时榜**：Subtask 1 = 0.7577，Subtask 2 = 0.4598，综合 = **0.6683**（第 6 名）。

模型来源：
- `rafalposwiata/deproberta-large-v1` — 领域专家，基于 r/depression + r/SuicideWatch 语料预训练，与赛题数据同源（LT-EDI-ACL2022 抑郁检测冠军方案 OPI 同款）。
- `roberta-large` — 通用强基线。

> 关键提分点：Subtask 2 的**逐类阈值校准**相比统一阈值 0.5 提升 +14.5 个百分点（救活 SORI、EOS 等稀有因子）；Subtask 1a 融合相比最强单模型提升 +0.68 个百分点（两模型互补，198 帖一错一救）。

---

## 数据与预处理

- 原始 `train.xlsx`：1635 帖 / 153 唯一用户（用户级时序数据）。
- 预处理脚本 `data_preprocess/preprocess.py`（`SEED=42`，可复现），产物 `data/train_clean.csv`（1635 行 × 32 列）。

修复的 4 项数据质量问题：
1. **风险标签大小写/空格不一致** → `strip().lower()` 归一化为 4 类。
2. **CV 必须按用户分组** → 否则同用户帖跨训练/验证集，OOF 虚高。采用 **StratifiedGroupKFold**（按 `anon_user_id` 分组 + 按 `risk_level` 分层，5 折），断言无用户跨折。
3. **factors 字段重复标签**（812 行）→ `ast.literal_eval` + 去重 → 24 列 `f_*` 0/1 标签。
4. **evidence 缺失**（5 行）→ `evidence_missing` 布尔列标记，不删除。

> 三任务、所有模型**共用同一份 `train_clean.csv` 与同一套 fold 划分**，保证 OOF 对比公平、可复现。
> 已知限制：极稀有因子（SORI 仅 8 正样本、EOS 仅 14）无法在各折均衡，OOF 方差较大，结论需谨慎解读。

---

## 目录结构

```
.
├── data_preprocess/
│   └── preprocess.py              # 数据清洗 + 固定 5-fold 划分
├── data/
│   ├── train.xlsx                 # 原始训练集
│   ├── train_clean.csv            # 预处理产物（含 fold + 24 因子列）
│   └── leaderboard.xlsx           # test 集（378 帖）
├── version_lynn/v1/
│   ├── common.py                  # 共享地基：数据加载 / fold / 评测 / 模型加载
│   ├── task1a_Dep55_Roberta45/    # task1a 五折训练 + 融合分析
│   ├── task1b_train.ipynb         # task1b BIO 证据抽取
│   ├── task2_train.ipynb          # task2 多标签训练 + 阈值校准
│   ├── make_submission.ipynb      # 全量训练 + 预测 test + 生成提交 csv
│   └── results/                   # OOF 概率缓存(.npz)、分数汇总(.csv)、混淆矩阵
├── technical_report/              # 启动文档、数据预处理记录、负结果记录
├── dev_record/                    # 每日进展记录
└── reference/                     # 相关论文 PDF
```

---

## 复现流程

```bash
# 1. 数据预处理（依赖缺失会自动 pip 安装）
cd data_preprocess
python3 preprocess.py path/to/train.xlsx   # 产出 train_clean.csv

# 2. 各子任务五折训练 + OOF 评估（Jupyter）
#    version_lynn/v1/ 下的 task1a / task1b / task2 notebook
#    OOF 概率与阈值缓存到 results/*.npz

# 3. 生成提交：全量 train 重训 → 预测 test → 按官方格式生成 csv
#    version_lynn/v1/make_submission.ipynb
#    融合权重、task2 逐类阈值均沿用 OOF 上确定的配置
```

---

## 方法论原则（NLPCC 验证有效，已迁移）

- **5-fold CV + OOF 评估**：固定 fold，所有模型共用，公平对比。
- **只信 OOF、不盲刷 public**：隐藏集最终评 + 每天限 3 次提交，OOF 是制胜关键。
- **逐类阈值校准**：task1a（针对 Weighted F1 重搜，**不照搬** Macro 系数）+ task2（multi-label 逐类）。
- **混淆矩阵驱动错误分析**：定位相邻风险等级的混淆模式。
- **互补性验证后再融合**：理论互补 ≠ 实际涨分，融合/校准改动一律 OOF 验证。
- **负结果也是成果**：系统记录被否决的方法（见下），体现工作彻底性，报告加分。

---

## 已记录的关键决策与负结果

详见 `technical_report/`：

- **模型选型三轮演化**：`MentalRoBERTa`（门控仓库受阻）→ `DepRoBERTa`；移除 `BERTweet`（max_len=128，实测 45% 长帖证据、10.6% 帖关键证据落在 128 token 后，截断不可接受）。
- **放弃 DeBERTa-v3-large（负结果）**：排查并修复六类技术故障后训练仍退化到多数类（Weighted F1 锁死 0.2024），判定该模型 + 本数据 + 环境组合不收敛，改用 RoBERTa-large。详见 `DeBERTa排查记录_负结果.md`。

---

## 下一步方向

Subtask 1 已接近头部（榜内第 4 高），**Subtask 2 是明显短板**（0.46 vs 头部 0.55~0.61），是拉开差距的主因。优先级：

1. **task2 融合**：加入 RoBERTa 一起概率层融合（参考 task1a 涨分经验）。
2. **稀有类处理**：对 F1≈0 的稀有因子用加权 BCE / focal loss / 过采样。
3. **调超参 / 增训练轮数**。

> 性价比测算：task2 从 0.46 → 0.55，综合分 +0.027 → 约 0.695，可冲第 3-4 名；task1 边际收益仅约 +0.01。
