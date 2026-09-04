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
│   ├── prepare_data.py
│   └── reward.py
├── scripts/
│   ├── build_test_parquet.sh
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
{"ranking":["当归","川芎","白芍"]}
```

输出必须是候选全集的一个排列，不能增加、删除或重复中药。因此本方法不进行处方长度
预测，也不包含 length reward。

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

默认输出：

```text
data/processed/test_top50.jsonl
data/processed/test_top50.parquet
```

脚本默认从 Top-200 中截取前 50 味作为待重排候选。使用全部 200 味时：

```bash
CANDIDATE_K=200 bash scripts/build_test_parquet.sh
```

重排 200 味时，应同步设置 `MAX_RESPONSE_LENGTH=2048`。当前测试源数据中，
Top-50/100/200 的 GT micro recall 分别约为 0.6754、0.8204 和 0.9274。

对齐过程会强制检查：

- 两个文件的有效样本数完全一致；
- 每一行的症状 ID 完全一致；
- 所有候选中药 ID 均能映射到药名；
- 同一病例的候选 ID 不重复；
- 症状、文本和 GT 字段完整。

测试脚本默认保留候选集与 GT 无交集的病例，避免评测偏高。构建训练集时应使用同一个
Python 模块并设置 `--unreachable-policy drop`，因为此类训练样本没有排序学习信号。

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

## 5. 候选集归一化 NDCG

设候选集为 $C$，真实药方为 $Y^*$，可达 GT 为：

$$
R=Y^*\cap C.
$$

所有 NDCG 只使用 $R$。因此，GNN 没有召回的真实中药不会错误地惩罚 reranker。

$$
\mathrm{NDCG@h}=\frac{
\sum_{i=1}^{\min(h,K)}\frac{\mathbb{I}[p_i\in R]}{\log_2(i+1)}
}{
\sum_{i=1}^{\min(h,|R|)}\frac{1}{\log_2(i+1)}
}.
$$

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
- 漏项、重复项、非字符串元素和候选外中药都会降低奖励；
- 完整合法且排序理想时，总奖励为 1。

奖励函数额外返回 `ndcg_5/10/20`、`gate_5/10`、`candidate_coverage`、
`exact_permutation` 等指标，供 VERL 日志记录和实验分析。

## 9. 安装与数据准备

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

## 10. 启动训练

### 10.1 开启能力门控：本文方法

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-0.6B \
HIERARCHICAL_REWARD=on \
bash scripts/train.sh
```

### 10.2 关闭能力门控：固定联合奖励

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

### 10.3 使用 Qwen3-1.7B

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-1.7B \
HIERARCHICAL_REWARD=on \
bash scripts/train.sh
```

### 10.4 单卡显存不足

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

## 11. 测试

奖励函数仅依赖 Python 标准库，可以脱离 VERL 单独验证：

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/*.sh
```

当前测试覆盖：门控公式、固定奖励、零奖励死区、开关公平性、候选外 GT、非法 JSON、
漏药、重复药、候选外药、非字符串元素、Qwen `<think>` 包装和 GT 提示词泄漏。

## 12. 方法边界

- 本方法是 reranker，不能找回 GNN 未召回的中药；论文中应单独报告候选 Recall@K。
- 候选顺序必须固定来自同一个 retriever，方法与消融不能使用不同候选集。
- `HIERARCHICAL_REWARD=on/off` 的实验应使用不同实验名，但保持相同随机种子、数据划分、
  rollout 数量和训练步数。
