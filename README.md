# Herb Reranker with VERL/GRPO

一个不修改 VERL 源码的中药候选重排工程。模型读取症状列表、原始症状描述和
GNN Top-K 候选中药，从候选中输出固定 Top-20 排序。奖励以 GNN 为零基线：复制为
0、改善为正、退化为负，避免模型通过原样复述召回结果获得高分。

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
│   ├── reward.py
│   └── validate_parquet.py
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

序号采用 1-based 候选 ID。默认从 50 个候选中输出恰好 20 个互不重复的 ID；它们
决定论文所需的全部 `@5/10/15/20` 指标，未输出的尾部候选不参与训练。需要完整排序时，
可以在推理后把剩余候选按原 GNN 顺序补到尾部。这是固定 Top-20 reranking，不是处方
长度预测，也不包含 length reward。

候选 ID 使用由 `sample_id` 决定的可复现置换，不等于 GNN 名次。提示词会同时展示
“候选ID”和“GNN名次”。这样 `[1,2,...,20]` 不再天然等于复制 GNN；模型若要保留
GNN 顺序，也必须读取候选内容和名次。原始 GNN 顺序单独存入 `gnn_candidate_herbs`，
因此 baseline 评测不会受到 ID 置换影响。

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

默认输出固定 Top-20；如需做输出规模消融，可以显式设置 `OUTPUT_K`，但同一组实验的
训练、验证和测试必须保持一致：

```bash
OUTPUT_K=20 bash scripts/build_train_parquet.sh
OUTPUT_K=20 bash scripts/build_test_parquet.sh
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
        "candidate_herbs": [...],       # 按候选ID索引的顺序
        "gnn_candidate_herbs": [...],   # GNN原始顺序
        "candidate_k": 50,
        "output_k": 20,
        "candidate_id_scheme": "deterministic_permuted_v1",
        "retriever": "gnn",
    },
}
```

提示词会直接写入 Parquet。若从旧版“输出药名”或“完整输出50个连续序号”协议升级，
必须重新构建 train/test Parquet；若还使用测试相关训练子集，也要在重建后重新执行
筛选。旧 checkpoint 与新的候选 ID 和 Top-20 协议不兼容，不能恢复到新实验中。

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

先分别计算模型与原始 GNN 的层级（或固定联合）排序分数 $S_M,S_B\in[0,1]$，
再把相对改善按当前病例的可用空间归一化：

$$
R_{\mathrm{imp}}=
\begin{cases}
\dfrac{S_M-S_B}{1-S_B+\epsilon}, & S_M\ge S_B,\\
\dfrac{S_M-S_B}{S_B+\epsilon}, & S_M<S_B.
\end{cases}
$$

因此复制 GNN 固定为 0，改善为正，退化为负。额外的安全防复制奖励为：

$$
R_{\mathrm{ac}}=\lambda\max(R_{\mathrm{imp}},0)
\left(1-\mathrm{CopyRatio@20}\right).
$$

它只放大“已经改善且确实改变位置”的结果；更差的随机排序不会因为与 GNN 不同而
获得奖励。合法输出的总奖励为：

$$
R=0.95R_{\mathrm{imp}}+0.05R_{\mathrm{format}}+R_{\mathrm{ac}}.
$$

输出必须恰好包含 `output_k=20` 个不同的合法整数 ID。非法 JSON、数量错误、重复、
非整数或越界输出不会获得排序奖励，其分数被放在所有合法输出的理论下界以下；分级
格式分数只用于区分训练早期不同程度的格式错误。

Precision 和 Recall 已由 reward 函数计算并返回，但默认不再叠加进标量 reward。
原因是固定 cutoff 下二者都主要由命中数量决定，直接与二元 NDCG 相加会重复计权并
削弱对头部位置的敏感性。标量 reward 仍由层级 NDCG 与格式约束组成。

奖励函数还返回 `relative_improvement`、`anti_copy_bonus`、`valid_output`、
`exact_topk`、`copy_ratio_output`、`duplicate_index_count`、`invalid_index_count`、
`missing_index_count` 和 `extra_index_count`，便于直接定位退化或协议错误。

## 9. 每 N 步监控测试集重排效果

`VAL_FILES` 指定测试 Parquet，`TEST_FREQ=N` 控制每 N 个训练 step 进行一次确定性
greedy 验证。验证只生成 1 个排序，避免采样噪声：

```bash
VAL_FILES=data/processed/test_top50.parquet \
TEST_FREQ=20 \
bash scripts/train.sh
```

每次验证都会在 cutoff `5/10/15/20` 上记录以下指标：

| SwanLab 变量名 | 含义 |
|---|---|
| `gnn_precision_k`, `gnn_recall_k`, `gnn_ndcg_k` | GNN 原始候选顺序；训练中应保持不变 |
| `model_precision_k`, `model_recall_k`, `model_ndcg_k` | 当前模型重排结果 |
| `oracle_precision_k`, `oracle_recall_k`, `oracle_ndcg_k` | 固定候选集内把所有可达 GT 前置后的理论上限 |
| `delta_precision_k`, `delta_recall_k`, `delta_ndcg_k` | `model - gnn`；大于 0 表示重排改善 |
| `headroom_precision_k`, `headroom_recall_k`, `headroom_ndcg_k` | `oracle - gnn`；当前 retriever 留给 reranker 的提升空间 |
| `remaining_gap_precision_k`, `remaining_gap_recall_k`, `remaining_gap_ndcg_k` | `oracle - model`；模型尚未实现的提升空间 |
| `copy_ratio_k`, `exact_copy_k` | 前 k 个位置复制 GNN 的比例，以及是否完整复制 |

同时返回 `gnn_rank_score`、`rank_delta=rank_score-gnn_rank_score` 和归一化后的
`relative_improvement`。主 reward 使用相对值来统一不同难度病例的尺度，并让复制行为
在监控中明确对应 0；真正的 GRPO 学习信号仍来自同一病例多个 rollout 之间的质量差异。

基于三个端到端均值，可以在论文中报告候选上限归一化提升：

$$
\mathrm{NormalizedGain@k}=
\frac{\overline{M@k}-\overline{B@k}}
{\overline{O@k}-\overline{B@k}+\epsilon},
$$

其中 $B$、$M$、$O$ 分别表示 GNN、Model 和 Oracle。应先在测试集上分别求均值再
计算该比值，不要在单病例上计算比值后取平均，以免小 headroom 病例放大噪声。

`copy_ratio_k` 本身不作为负向 penalty。它只在模型已经优于 GNN 时调节一个很小的
正向 bonus，从而避免迫使本来已经正确的 GNN 排序做无意义交换。

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

首次使用本版本前必须重新构建 Parquet。训练入口会逐条检查 `candidate_id_scheme`、
`output_k`、GNN 原顺序和 Prompt 版本；若误用了旧 Parquet，会在启动 Ray 前直接报错。
训练脚本默认 `RESUME_MODE=disable`，实验名中
包含 `top20-relative-v2`，用于隔离旧的药名/完整排列 checkpoint。只有继续同一协议的
中断任务时，才显式设置 `RESUME_MODE=auto`。

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
checkpoints/Qwen3-0.6B-top20-relative-v2-hierarchical/
checkpoints/Qwen3-0.6B-top20-relative-v2-fixed/
```

避免两个实验错误地恢复彼此的 checkpoint。

### 11.3 使用 Qwen3-1.7B

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-1.7B \
HIERARCHICAL_REWARD=on \
RESUME_MODE=disable \
bash scripts/train.sh
```

训练默认使用 `ROLLOUT_N=8`、`ROLLOUT_TEMPERATURE=1.2`、`ROLLOUT_TOP_P=0.95` 和
`ENTROPY_COEFF=0.005` 保留探索。若新运行的同一病例仍产生完全相同的8条输出，应先做
短程 Top-20 SFT warm-up，而不是继续提高 copy penalty。

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

当前测试覆盖：门控公式、固定奖励、相对改善归一化、复制零基线、安全防复制 bonus、
固定 Top-20、候选 ID 置换、非法 JSON、漏项、重复/越界/字符串序号、Qwen `<think>`
包装、Oracle 上界、GNN headroom、remaining gap、完整指标差值和 GT 提示词泄漏。

## 13. 方法边界

- 本方法是 reranker，不能找回 GNN 未召回的中药；论文中应单独报告候选 Recall@K。
- 候选顺序必须固定来自同一个 retriever，方法与消融不能使用不同候选集。
- `HIERARCHICAL_REWARD=on/off` 的实验应使用不同实验名，但保持相同随机种子、数据划分、
  rollout 数量和训练步数。
