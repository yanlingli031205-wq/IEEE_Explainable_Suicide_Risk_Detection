# DeBERTa-v3-large 微调排查记录（负结果）
# IEEE BigData 2026 Explainable Suicide Risk Detection — Subtask 1a

> 用途：完整记录 DeBERTa-v3-large 作为候选模型的选型理由、微调过程中遭遇的一系列技术障碍、逐一排查与修复、以及最终放弃的决策依据。供撰写 solution report 的 "Model Selection" 与 "Negative Results / Lessons Learned" 章节使用。
> 结论先行：DeBERTa-v3-large 在本数据 + Colab(Pro) 环境下，技术故障可逐一修复，但训练始终无法收敛（退化到多数类，Weighted F1 锁定 0.2024）。最终改用 RoBERTa-large 作为第二模型。

---

## 一、为什么选 DeBERTa-v3-large

模型池设计遵循"互补性"原则：三类视角各取其一。

| 角色 | 模型 | 理由 |
|---|---|---|
| 领域专家 | DepRoBERTa-large | RoBERTa-large 在 r/depression + r/SuicideWatch 语料上继续预训练，与本赛数据同源 |
| 通用强者 | **DeBERTa-v3-large** | 架构创新（解耦注意力 disentangled attention + ELECTRA 式预训练），通用 NLU 任务常为 SOTA |
| 经典对照 | RoBERTa-large | 标准强基线 |

选 DeBERTa 的核心动机是**互补性**：它与 DepRoBERTa 架构不同、领域不同，理论上错误模式差异大，是融合的理想搭档。DepRoBERTa 单模型已稳定达到 OOF Weighted F1 0.79~0.82，若能引入一个架构迥异的强模型，融合有较大涨分空间。

---

## 二、遭遇的技术障碍与逐一排查

DeBERTa-v3 的微调以"难搞"著称，本项目依次遭遇并修复了以下六类问题。每一项均有明确现象、根因与对策。

### 障碍 1：FP16 梯度反缩放失败
- **现象**：`ValueError: Attempting to unscale FP16 gradients`，训练第一步即崩。
- **根因**：DeBERTa-v3 与 fp16 混合精度不兼容；disentangled attention 产生的梯度幅度较小，fp16 的梯度缩放器处理异常。
- **对策**：关闭 fp16，改用 bf16（后续发现 bf16 亦有问题，见障碍 5）。

### 障碍 2：学习率过高导致训练发散
- **现象**：沿用 DepRoBERTa 的 lr=2e-5 时，training loss 第 1 epoch 飙至 3.49、validation loss 至 3.88（正常四分类初始约 1.39），模型退化。
- **根因**：DeBERTa-v3-large 对学习率极度敏感，2e-5 量级偏高。
- **对策**：按模型分设学习率，DeBERTa 降至 5e-6 ~ 1e-5。

### 障碍 3：Loss 数值爆炸
- **现象**：降学习率后某次 validation loss 爆至 78.71（天文数字），Weighted F1 跌至 0.0099。
- **根因**：DeBERTa-v3 微调的已知数值不稳定问题（社区多次报告：训练中途 loss 突然 spike、梯度范数变 NaN）。
- **对策**：全程 fp32 + 梯度裁剪 max_grad_norm=1.0 + 延长 warmup 至 0.2。爆炸消除，loss 趋于平稳（约 1.27）。

### 障碍 4：分类头 pooler 缺失（关键）
- **现象**：loss 已平稳下降，但 Weighted F1 锁死在 0.2024（即"全预测多数类 indicator"的得分）。模型加载报告显示 `pooler.dense.weight | MISSING`。
- **根因**：`AutoModelForSequenceClassification` 加载 DeBERTa-v3 时，默认分类流程依赖一个 pooler 层；但 DeBERTa-v3 预训练权重中不含 pooler，导致该层随机初始化且未被正确接入，传入 classifier 的特征无效。
- **对策**：弃用默认分类封装，自定义 `DebertaClassifier`：加载 `AutoModel` 主干 → 取 [CLS] 位置表示（last_hidden_state[:, 0, :]）→ dropout → linear 输出。绕开 pooler。加载报告中 pooler 消失，确认修复生效。

### 障碍 5：主干与分类头精度不匹配
- **现象**：`RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`。
- **根因**：自定义类中主干输出为半精度，而新增 linear 分类头为 float32，前向相乘类型冲突。
- **对策**：forward 中显式对齐 `cls = cls.to(self.classifier.weight.dtype)`，自适应任意精度。

### 障碍 6：token_type_ids 干扰
- **现象**：以上全部修复后，训练技术上完全跑通（4 epoch 无报错，loss 平稳 1.27），但 Weighted F1 仍锁死 0.2024，模型持续退化到多数类。
- **根因（有文献背书）**：DeBERTa-v3 不使用 segment IDs（token_type_ids），而自定义 forward 仍将 tokenizer 输出的 token_type_ids 传入主干，对模型造成干扰。（参见 AI 文本检测 benchmark 论文实测记录：DeBERTa-v3 微调须省略 token_type_ids，且 bf16 会因 disentangled attention 的小梯度而"静默归零"，fp16 导致缩放不稳定，故须全程 fp32。）
- **对策**：forward 中不再向主干传 token_type_ids（签名保留以兼容 Trainer）；同时确认全程 fp32；学习率回调至 1e-5。

---

## 三、最终结果与放弃决策

经上述六轮修复，DeBERTa-v3 在技术层面已**完全可运行**：无报错、loss 平稳下降。但**模型始终无法学到有效区分**，Weighted F1 在所有 fold、所有 epoch 上恒定为 0.2024 / 0.2035（即多数类基线），与 DepRoBERTa 的 0.79~0.82 形成断崖差距。

判断：这已非单点 bug，而是 **DeBERTa-v3-large + 本数据规模(1635 帖) + Colab 环境** 这一特定组合下的训练不收敛。可能成因包括 disentangled attention 在 fp32 下对超参的高敏感性、小数据量下的优化困难等，继续调参的边际收益与时间成本严重失衡（已耗约八轮迭代）。

**决策**：放弃 DeBERTa-v3-large，改用 **RoBERTa-large** 作为第二模型。依据有三：
1. RoBERTa-large 为标准架构，无上述任何坑，2e-5 + bf16 直接稳定收敛。
2. **有冠军先例**：LT-EDI-ACL2022 抑郁检测共享任务冠军方案（OPI）正是 **DepRoBERTa + RoBERTa-large 集成**取得第一，证明二者虽同架构但因领域预训练差异而互补。
3. 时间约束：报名与提交临近，需将资源投入可推进的方案。

---

## 四、写进报告的价值（Negative Results）

本节是一段完整、可复现的技术叙事，体现排查的系统性：
- 选型有据（互补性原则）
- 故障分类清晰（6 类，各有现象/根因/对策）
- 根因定位有文献支撑（非臆测）
- 放弃决策有量化依据（0.2024 多数类基线 vs 0.8）与替代方案背书（冠军组合）

在"创新 40% + 报告 30%"的评分结构下，这种诚实而彻底的负结果记录是加分项，区别于仅罗列成功配置的浅层报告。

---

## 五、可复用的经验（Lessons Learned）

针对未来使用 DeBERTa-v3 微调，汇总避坑清单：
1. 全程 **fp32**（fp16 报 unscale 错误；bf16 静默归零梯度）。
2. 学习率取 **1e-5 量级**，配 10%~20% warmup 与 cosine/linear 调度。
3. 加 **梯度裁剪** max_grad_norm=1.0。
4. **省略 token_type_ids**（DeBERTa-v3 不用 segment IDs）。
5. 分类头**自定义**取 [CLS] 表示，勿依赖默认 pooler（pooler 在预训练权重中缺失）。
6. 即便以上全做对，小数据场景仍可能不收敛——应设"止损轮数"，及时切换稳定基线（如 RoBERTa-large）。
