# 中药候选重排：旧版指标与渐进式 GRPO

以旧版 reward 的药名输出、GNN 相对增益和天花板保持逻辑为基础，统一数据、训练奖励和离线评价。运行入口是 `herb_reranker/reward.py:compute_score`，不是根目录下的历史 reward 文件。

## 本次协议

- 输入保留规范症状列表、**原始症状描述**、GNN 原始候选药名顺序，不再使用治法，也不把同症状病例合并成一条。
- 输出 `<answer>药名1>药名2>...</answer>`，至少 20 味互不重复的候选药，建议仅输出 20 味。允许更多合法药名，评价只取前 20。
- 可选 `<think>` 简洁说明，提示不超过 200 字；内容与长短不提供质量奖励。
- 不输出候选 ID，不置换 ID，不做别名映射，不用 GNN/GT 自动补齐模型答案。
- 训练默认丢弃无可达 GT 病例，验证和测试保留。原始数据、预测、Parquet 和 checkpoint 不上传 Git。

## 指标与 Reward

所有指标来自同一个 `ranking_metrics` 函数：

- 先按药名保序去重，再截取前 20；候选外 GT 不能被输出直接命中。
- P@K = H@K / K，R@K = H@K / 完整 GT 数，F1 使用二者的调和平均。
- NDCG 使用旧 `metric.py` 的 `batch_test` 口径：`method=1, k_max=20`。IDCG 对**预测前 20 的相关性向量**排序后计算，不改为标准全 GT IDCG。
- 因此补入深位 GT 可能改变 NDCG@5 的分母。头部保护使用命中数，而不是把这个 NDCG 当作不可退步约束。

固定一条病例及 K 后，P/R/F1 都随命中数 H 单调变化。无需再人为分配 P、R、F1 的权重。

| 阶段   | 按优先级比较的目标                         |
| ------ | ------------------------------------------ |
| stage1 | H5，然后 NDCG5                             |
| stage2 | 最小化 D5，然后 H10，然后 NDCG10           |
| stage3 | 最小化 D5，然后 D10，然后 H20，然后 NDCG20 |

`Dk = max(0, 上一阶段冻结模型的 Hk - 当前 Hk)`。Stage 2/3 必须提供上一阶段逐病例预测，不允许静默回退为 GNN 参考。

整数优先级用可达命中数确定的混合进制编码：从 `NDCG / 2` 开始，逐级计算 `(当前整数 + 低优先级尾项) / (该项最大值 + 1)`。低优先级尾项始终小于 1，因此任何一个头部命中的损失都不能靠后部指标补偿。这是优先级规则，不是手调的 Top-5/10/20 加权和。

最终保留旧版可解释结构：

1. 普通质量项 = 当前阶段编码分数 - GNN 的同口径分数。
2. **合法完整答案**达到当前阶段可达天花板且存在可达 GT 时，质量项为 1.0，允许学习保持/复制。
3. 无可达 GT 不触发天花板奖励。
4. 合法格式奖励为 0.1；重复、候选外药、有效药名不足的每项处罚为 0.03，总处罚封顶 1。
5. 非法/不足量答案保留原始观测指标，但质量项不允许为正，不能靠短答案拿最优奖励。

这里仍保留格式系数；移除的是手调的**多 cutoff 质量权重**。不再保留旧版正负增益的不对称缩放，避免反转合法答案的优先级。

注意：GRPO 同组减去相同 GNN baseline 会在中心化中抵消，不能把它本身当成防复制机制。真正的排序信号来自命中数/NDCG、冻结头部参考及合法性约束。Reward 的优先级也不等于优化器对最终模型给出“不退步”的数学保证，仍须检查验证集的 P/R/F1 和 `reference_deficit_5/10`。

## 构建数据

```bash
pip install -r requirements.txt
```

原始 JSONL 每行包含 `sample_id`、`symptoms`、`symptom_description`、`candidate_herbs`、`ground_truth_herbs`。

也支持从上下文与 GNN 输出合并：

```bash
OUTPUT_DIR=data/processed/source bash scripts/build_train_parquet.sh
bash scripts/build_test_parquet.sh
```

默认输入为 `data/raw/{train,test}_with_context.jsonl` 和 `data/raw/{train,test}_candidate_name.txt`。候选文件每行是 `症状名称列表<TAB>有序药名列表`。

- 优先按症状顺序做子序列对齐：候选文件可比上下文少一些病例，重复症状仍使用各自对应行的候选顺序，不再强行压成字典。
- 若被跳过病例的症状在候选文件中也出现，则不能确定是哪次重复病例缺失；只有候选顺序无冲突时才允许按唯一症状映射回退，否则报错。
- 训练缺少候选的病例单独计入 `dropped_missing_candidate_rows`；测试/验证使用 `keep` 时若缺候选会报错，不静默缩小评测集。
- 可用 `RETRIEVAL_FILE=/path/gnn_ids.txt HERB_MAPPING=/path/herb_mapping.txt` 切换到旧 ID 文件输入；此模式严格逐行校验症状 ID，再把药材 ID 映射成药名。

原始输入与候选文件不对齐时必须先修正来源，不能按行盲目拼接。相同症状的不同原始描述仍是独立病例。

从训练集划分验证集，最终测试集不参与 checkpoint 选择：

```bash
python -m herb_reranker.split_data \
  --input data/processed/source/train_top50.jsonl \
  --train-output data/processed/splits/train.jsonl \
  --validation-output data/processed/splits/val.jsonl

python -m herb_reranker.prepare_data \
  --input data/processed/splits/train.jsonl \
  --output data/processed/stage1/train_top50.parquet
python -m herb_reranker.prepare_data \
  --input data/processed/splits/val.jsonl \
  --output data/processed/stage1/val.parquet --unreachable-policy keep
```

划分只使用原始输入和固定 seed，相同输入组不跨训练/验证集。默认验证组比例为 10%，不按 GT 分层或筛选。数据中所有 `sample_id` 必须唯一。

## 按测试症状筛选训练子集

原来的筛选代码仍然保留：`herb_reranker/filter_train_by_test.py` 和 `scripts/filter_train_by_test.sh`。只使用测试输入的 `extra_info.symptoms` 决定选择，不使用测试 GT、reward、候选排序或上一阶段参考。

```bash
TRAINING_STAGE=stage1 FILTER_MODE=matched TRAIN_PER_TEST=2 \
  bash scripts/filter_train_by_test.sh

TRAINING_STAGE=stage1 \
  TRAIN_FILES=data/processed/stage1/train_top50_test_matched.parquet \
  MODEL_PATH=/path/base-model VERL_ROOT=/path/verl bash scripts/train.sh
```

默认 `matched` 按测试症状组合分配训练病例，缺少精确组合时可按 Jaccard 相似度兜底。`FILTER_MODE=exact` 只保留精确症状组合，`FILTER_MODE=overlap` 保留任一症状有交集的病例。`TRAIN_PARQUET`、`TEST_PARQUET`、`OUTPUT_PARQUET` 可显式覆盖路径；旧版非阶段目录仍可读取。筛选保留整行 Arrow schema，因此原始描述和 Stage 2/3 的参考不会丢失。

这是使用测试输入分布的开发/传导式实验选项，不是默认的独立留出评测流程；正式比较应明确报告筛选协议，并同时报告完整训练集结果。不得用测试 GT 选择病例或 checkpoint。

## 逐阶段训练

```bash
TRAINING_STAGE=stage1 MODEL_PATH=/path/base-model \
  VERL_ROOT=/path/verl bash scripts/train.sh
```

默认训练文件为 `data/processed/{stage}/train_top50.parquet`，验证文件为同目录 `val.parquet`。也可明确传入 `TRAIN_FILES` 和 `VAL_FILES`。不再默认把测试集当验证集。训练前会检查药名协议、指标口径、阶段、输入指纹和参考。

Stage 1 完成后，根据**验证集**选择 checkpoint，导出为可推理的模型并部署本地兼容 `/v1/chat/completions` 的服务。以下以服务模型名 `stage1-selected` 为例：

```bash
python -m herb_reranker.predict \
  --data data/processed/splits/train.jsonl \
  --output outputs/stage1/train.pred.jsonl --model stage1-selected
python -m herb_reranker.evaluate \
  --data data/processed/splits/train.jsonl \
  --predictions outputs/stage1/train.pred.jsonl \
  --reference-output outputs/stage1/train.ref.jsonl --checkpoint-id stage1-selected
```

对验证集同样预测并导出 `val.ref.jsonl`。推理默认访问 `http://127.0.0.1:8000/v1`，可用 `--base-url` 更改。只发送 Prompt，不发送 GT 或参考；预测记录带输入指纹。参考导出要求所有答案合法且至少 20 味，不自动补齐非法输出。

随后构建 Stage 2：

```bash
python -m herb_reranker.prepare_data \
  --input data/processed/splits/train.jsonl \
  --output data/processed/stage2/train_top50.parquet \
  --training-stage stage2 --reference-jsonl outputs/stage1/train.ref.jsonl
python -m herb_reranker.prepare_data \
  --input data/processed/splits/val.jsonl \
  --output data/processed/stage2/val.parquet --unreachable-policy keep \
  --training-stage stage2 --reference-jsonl outputs/stage1/val.ref.jsonl

TRAINING_STAGE=stage2 MODEL_PATH=/path/stage1-selected \
  VERL_ROOT=/path/verl bash scripts/train.sh
```

Stage 3 同理：用选定 Stage 2 模型生成新的训练/验证参考，构建 `stage3` Parquet，并从该模型初始化。阶段切换不是按固定 epoch 自动触发，必须确认验证集头部没有明显退步；这避免把错误的自动阈值隐藏在脚本中。

默认关闭自动恢复，防止继续使用旧 ID 协议 checkpoint 的训练状态。模型名输出默认 response 上限 512 tokens，可通过 `MAX_RESPONSE_LENGTH` 覆盖。

## 最终评测

用选定最终模型预测完整测试 JSONL，再执行：

```bash
python -m herb_reranker.evaluate \
  --data data/processed/stage1/test_top50.jsonl \
  --predictions outputs/final/test.pred.jsonl
```

返回模型与 GNN 的 P/R/F1/NDCG@5/10/20、差值、合法输出率、无可达 GT 比例。非法答案的实际排序指标也保留，但必须同时报告合法率，不能只筛选成功输出。测试集不需要提供训练用的上一阶段参考。

## 验证与边界

```bash
python -m unittest discover -s tests -v
```

覆盖旧指标数值、天花板复制、无可达 GT、词典外药、短答案、阶段优先级、候选对齐、原始描述保留、参考指纹、完整 Parquet 往返和模拟推理。测试不需要 GPU，也不访问网络。

实现和离线测试不能替代真实 GRPO 对照实验。是否恢复或超过旧版约 4% 的提升，需要在相同候选、拆分、模型和训练预算下重新比较。
