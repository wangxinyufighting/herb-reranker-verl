# Herb Reranker with VERL/GRPO

一个不修改 VERL 源码的中药候选重排工程。模型读取症状列表、原始症状描述和
GNN Top-K 候选中药，输出候选全集的新排序。

当前实现支持两种严格可比的排序奖励：

- `HIERARCHICAL_REWARD=on`：能力门控的层级排序奖励，本文方法；
- `HIERARCHICAL_REWARD=off`：固定联合 NDCG 奖励，消融基线。

两种模式除排序奖励聚合方式外，数据、提示词、GRPO、格式约束和训练超参数完全一致。

## 1. 项目结构

```text
herb-reranker-verl/
├── data/
│   ├── train.example.jsonl
│   └── test.example.jsonl
├── herb_reranker/
│   ├── build_ptm_grpo_data.py
│   ├── filter_train_by_test.py
│   ├── prepare_data.py
│   └── reward.py
├── scripts/
│   ├── build_split_parquet.sh
│   ├── build_train_parquet.sh
│   ├── build_test_parquet.sh
│   ├── filter_train_by_test.sh
│   ├── prepare_data.sh
│   ├── train.sh
│   └── train_one_stage.sh
├── tests/
│   ├── test_prepare_data.py
│   └── test_reward.py
└── requirements.txt
```

`train.sh` 是用户启动入口；`train_one_stage.sh` 保存完整 VERL 参数，由前者调用。

## 2. 任务定义

输入：

1. 规范化症状列表；
2. 原始症状描述文本；
3. GNN 给出的有序 Top-K 候选中药。

模型只能重排候选中药，输出格式为：

```json
{"ranking":[3,1,2,4]}
```

序号采用 1-based 编号，`1` 对应候选列表中的第 1 味药。输出必须是 `1..K` 的完整
排列，不能增加、删除、重复序号，也不能直接输出药名。因此本方法不进行处方长度预测，
也不包含 length reward。序号协议显著缩短输出，并避免长药名引发的重复和截断错误。

## 3. 原始数据格式

训练、验证和测试数据均采用 UTF-8 JSONL，每行表示一个病例：

```json
{
  "sample_id": "train-000001",
  "symptoms": ["头痛", "乏力"],
  "symptom_description": "产后头痛，面色少华，神疲乏力。",
  "candidate_herbs": ["川芎", "当归", "白芍", "黄芪"],
  "ground_truth_herbs": ["当归", "川芎", "白芍"]
}
```

约束：

- `candidate_herbs` 保留 GNN 原始顺序，元素必须唯一；
- `ground_truth_herbs` 仅用于奖励和评测，不会进入模型提示词；
- `symptom_description` 必须来自诊疗时可获得的信息，不能根据真实药方反推；
- 训练集可丢弃候选集与 GT 完全无交集的样本；验证/测试集应保留，以免评测偏高。

### 3.1 合并 PTM 上下文与 GNN Top-200

`build_ptm_grpo_data.py` 读取三份文件：

```text
test_with_context.jsonl   症状列表、原始症状文本、GT 药方、症状 ID
test_top200_herbs.txt      GNN 输出：症状 ID + 200 个有序中药 ID
herb_mapping.txt          中药名称与 ID 的映射
```

建议将三份文件放到 `data/raw/`，然后执行：

```bash
bash scripts/build_test_parquet.sh
```

训练集使用完全相同的格式。准备以下文件：

```text
data/raw/train_with_context.jsonl
data/raw/train_top200_herbs.txt
data/raw/herb_mapping.txt
```

然后运行：

```bash
bash scripts/build_train_parquet.sh
```

默认得到 `data/processed/train_top50.parquet`。本次只提供了 test 的 GNN 输出，因此
训练构建代码已经就绪，但需要补充 `train_top200_herbs.txt` 后才能生成真实训练 Parquet。

默认输出：

```text
data/processed/test_top50.jsonl
data/processed/test_top50.parquet
```

脚本默认从 Top-200 中截取前 50 味作为待重排候选。使用全部 200 味时：

```bash
CANDIDATE_K=200 bash scripts/build_test_parquet.sh
```

序号输出下，Top-50 建议从 `MAX_RESPONSE_LENGTH=256` 开始，Top-200 建议从 1024
开始并根据 `response_length/clip_ratio` 调整。当前测试源数据中，
Top-50/100/200 的 GT micro recall 分别约为 0.6754、0.8204 和 0.9274。

对齐过程会强制检查：

- 两个文件的有效样本数完全一致；
- 每一行的症状 ID 完全一致；
- 所有候选中药 ID 均能映射到药名；
- 同一病例的候选 ID 不重复；
- 症状、文本和 GT 字段完整。

测试脚本默认保留候选集与 GT 无交集的病例，避免评测偏高。构建训练集时应使用同一个
Python 模块并设置 `--unreachable-policy drop`，因为此类训练样本没有排序学习信号。

### 3.2 构造测试症状相关的训练子集

开发阶段若完整训练集上的 GRPO 过慢，可以只根据测试集的输入症状构造较小训练集：

```bash
bash scripts/filter_train_by_test.sh
```

默认使用 `matched` 模式：将病例按完整症状组合分组，每个测试病例最多分配 2 条训练
病例；测试症状组合在训练集中不存在时，使用症状 Jaccard 最近邻兜底。筛选器只读取
测试集的 `extra_info.symptoms`，不会读取测试 GT、候选列表或奖励。输出为：

```text
data/processed/train_top50_test_matched.parquet
data/processed/train_top50_test_matched.report.json
```

训练时显式指定该文件：

```bash
TRAIN_FILES=data/processed/train_top50_test_matched.parquet \
  bash scripts/train.sh
```

通过 `TRAIN_PER_TEST` 控制规模，例如：

```bash
# 更快，约为每个测试病例保留 1 条训练病例
TRAIN_PER_TEST=1 bash scripts/filter_train_by_test.sh

# 更稳健，约为每个测试病例保留 3 条训练病例
TRAIN_PER_TEST=3 bash scripts/filter_train_by_test.sh
```

还支持两个不限制组内数量的诊断模式：

```bash
FILTER_MODE=overlap bash scripts/filter_train_by_test.sh
FILTER_MODE=exact bash scripts/filter_train_by_test.sh
```

注意：这是利用测试输入分布的 transductive 筛选。它适合快速调试，但正式论文实验应
同时报告完整训练集结果，或对所有方法采用完全相同的筛选协议。更严格的开发流程可以
将 `TEST_PARQUET` 指向验证集，以避免使用测试输入。

## 4. VERL Parquet 格式

`prepare_data.py` 将 JSONL 转换成 VERL 可读取的 Parquet：

```python
{
    "data_source": "ptm_herb_rerank",
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    "ability": "listwise_reranking",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"ground_truth_herbs": [...]},
    },
    "extra_info": {
        "sample_id": "...",
        "symptoms": [...],
        "symptom_description": "...",
        "candidate_herbs": [...],
        "candidate_k": 50,
        "retriever": "gnn",
    },
}
```

## 5. 评测指标

设候选集为 $C$，真实药方为 $Y^*$，可达 GT 为：

$$
R=Y^*\cap C.
$$

标量排序奖励中的 NDCG 只使用 $R$：

$$
\mathrm{NDCG@h}=\frac{
\sum_{i=1}^{\min(h,K)}\frac{\mathbb{I}[p_i\in R]}{\log_2(i+1)}
}{
\sum_{i=1}^{\min(h,|R|)}\frac{1}{\log_2(i+1)}
}.
$$

这使 GNN 没有召回的真实中药不会错误地惩罚 reranker，它衡量的是固定候选集内部的
排序能力。代码中旧键 `ndcg_5/10/15/20` 对应这套 reward 口径。

同时监控标准的端到端 Precision 和 Recall：

$$
\mathrm{Precision@h}=\frac{|P_{1:h}\cap Y^*|}{h},\qquad
\mathrm{Recall@h}=\frac{|P_{1:h}\cap Y^*|}{|Y^*|}.
$$

测试监控则使用标准的端到端 Precision、Recall 和 NDCG；三者的相关集合及 IDCG
都基于完整 $Y^*$：

$$
\mathrm{NDCG@h}_{\mathrm{test}}=\frac{
\sum_{i=1}^{\min(h,K)}\frac{\mathbb{I}[p_i\in Y^*]}{\log_2(i+1)}
}{
\sum_{i=1}^{\min(h,|Y^*|)}\frac{1}{\log_2(i+1)}
}.
$$

因此，测试 Recall 和 NDCG 都会反映 GNN 的候选召回上限；同一测试病例上，原始 GNN
与模型重排使用完全相同的分母，二者差值只反映排序变化。

记：

$$
s_5=\mathrm{NDCG@5},\quad
s_{10}=\mathrm{NDCG@10},\quad
s_{20}=\mathrm{NDCG@20}.
$$

## 6. 能力门控的层级排序奖励

开启 `HIERARCHICAL_REWARD=on` 时，使用软门控：

$$
g(s)=\epsilon+(1-\epsilon)s,
$$

以及统一的层级奖励：

$$
r_{\mathrm{rank}}^{\mathrm{hier}}
=\frac{
s_5+g(s_5)s_{10}+g(s_5)g(s_{10})s_{20}
}{3}.
$$

默认 $\epsilon=0.1$：

- Top-5 始终直接优化；
- Top-5 提高后，Top-10 自动获得更大权重；
- Top-5 和 Top-10 都提高后，Top-20 才被充分激活；
- 即使 Top-5 暂时为零，更深位置仍保留弱信号，不会形成奖励死区。

这是一种由当前排序能力驱动的样本级自节奏学习，不依赖 epoch、global step 或人工阶段。

## 7. 关闭能力门控的消融基线

设置 `HIERARCHICAL_REWARD=off` 后，使用固定联合奖励：

$$
r_{\mathrm{rank}}^{\mathrm{fixed}}
=0.4s_5+0.3s_{10}+0.3s_{20}.
$$

权重可通过 `FIXED_WEIGHT_5`、`FIXED_WEIGHT_10`、`FIXED_WEIGHT_20` 修改。

## 8. 完整奖励

两种模式共用相同的约束和格式奖励：

$$
r=0.95\,r_{\mathrm{rank}}q_{\mathrm{constraint}}
+0.05\,r_{\mathrm{format}}.
$$

- `q_constraint` 同时考虑候选合法率和候选覆盖率；
- `r_format` 提供很小的稠密格式反馈；
- 漏项、重复项、非整数或越界序号都会降低奖励；
- 完整合法且排序理想时，总奖励为 1。

Precision 和 Recall 已由 reward 函数计算并返回，但默认不再叠加进标量 reward。
原因是固定 cutoff 下二者都主要由命中数量决定，直接与二元 NDCG 相加会重复计权并
削弱对头部位置的敏感性。标量 reward 仍由层级 NDCG 与格式约束组成。

奖励函数还返回 `gate_5/10`、`candidate_coverage`、`exact_permutation` 等指标，供
VERL 日志记录和实验分析。

## 9. 每 N 步监控测试集重排效果

`VAL_FILES` 指定测试 Parquet，`TEST_FREQ=N` 控制每 N 个训练 step 进行一次确定性
greedy 验证。验证只生成 1 个排序，避免采样噪声：

```bash
VAL_FILES=data/processed/test_top50.parquet \
TEST_FREQ=20 \
bash scripts/train.sh
```

每次验证都会在 cutoff `5/10/15/20` 上同时记录三组指标：

| SwanLab 变量名 | 含义 |
|---|---|
| `gnn_precision_k`, `gnn_recall_k`, `gnn_ndcg_k` | GNN 原始候选顺序；训练中应保持不变 |
| `model_precision_k`, `model_recall_k`, `model_ndcg_k` | 当前模型重排结果 |
| `delta_precision_k`, `delta_recall_k`, `delta_ndcg_k` | `model - gnn`；大于 0 表示重排改善 |

在当前 VERL 中它们会显示为
`val-aux/ptm_herb_rerank/<变量名>/mean@1`。例如模型的 NDCG@10 是
`val-aux/ptm_herb_rerank/model_ndcg_10/mean@1`。GNN 指标也在每次验证时计算，
但只依赖固定测试数据，因此曲线应为水平线。

## 10. 安装与数据准备

先根据 VERL 官方说明安装其运行环境。本工程按照以下 VERL 提交核对接口：

```text
84e014b4c2ba2ec32c9a47fb85067454f90e8e82
```

安装额外数据依赖：

```bash
python3 -m pip install -r requirements.txt
```

把实际数据保存为 `data/train.jsonl` 和 `data/test.jsonl`，然后运行：

```bash
bash scripts/prepare_data.sh
```

正式论文实验建议再独立划分 validation/test，不使用 test 集调参。

## 11. 启动训练

### 11.1 开启能力门控：本文方法

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-0.6B \
HIERARCHICAL_REWARD=on \
bash scripts/train.sh
```

### 11.2 关闭能力门控：固定联合奖励

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-0.6B \
HIERARCHICAL_REWARD=off \
bash scripts/train.sh
```

两个模式默认分别写入：

```text
checkpoints/Qwen3-0.6B-grpo-hierarchical/
checkpoints/Qwen3-0.6B-grpo-fixed/
```

避免两个实验错误地恢复彼此的 checkpoint。

### 11.3 使用 Qwen3-1.7B

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-1.7B \
HIERARCHICAL_REWARD=on \
bash scripts/train.sh
```

### 11.4 单卡显存不足

```bash
VERL_ROOT=/path/to/verl \
HIERARCHICAL_REWARD=on \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_N=4 \
GPU_MEMORY_UTILIZATION=0.45 \
bash scripts/train.sh
```

## 12. 测试

奖励函数仅依赖 Python 标准库，可以脱离 VERL 单独验证：

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/*.sh
```

当前测试覆盖：门控公式、固定奖励、零奖励死区、开关公平性、候选外 GT、非法 JSON、
漏项、重复/越界/字符串序号、Qwen `<think>` 包装、完整指标差值和 GT 提示词泄漏。

## 13. 方法边界

- 本方法是 reranker，不能找回 GNN 未召回的中药；论文中应单独报告候选 Recall@K。
- 候选顺序必须固定来自同一个 retriever，方法与消融不能使用不同候选集。
- `HIERARCHICAL_REWARD=on/off` 的实验应使用不同实验名，但保持相同随机种子、数据划分、
  rollout 数量和训练步数。
