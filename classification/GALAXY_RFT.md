## Galaxy Visual-RFT

### Smoke test

```bash
bash /home/ruoqi/Visual-RFT/src/scripts/qwen3_vl_galaxy_smoke_test.sh
```

这条脚本适合先验证链路能否跑通。

### Full pipeline

```bash
bash /home/ruoqi/Visual-RFT/src/scripts/qwen25_vl_galaxy_sft_grpo.sh
```

这条脚本会依次完成：

1. 从 `/home/ruoqi/data/train.csv` 和 `/home/ruoqi/data/test.csv` 生成更强的 Visual-RFT 数据集
2. 先做 `Qwen2.5-VL-7B-Instruct` 的 SFT 预热
3. 再在 SFT 模型上做 GRPO 微调
4. 最后进行带 prompt voting 的评估，并输出整体准确率、per-class accuracy、confusion matrix、错例分析

### 单独构建数据

```bash
python3 /home/ruoqi/Visual-RFT/classification/prepare_galaxy_rft_dataset.py \
  --train_csv /home/ruoqi/data/train.csv \
  --test_csv /home/ruoqi/data/test.csv \
  --output_dir /home/ruoqi/Visual-RFT/share_data/galaxy_rft_custom \
  --augment_train
```

输出内容包括：

- `DatasetDict`
- `train_sft.jsonl`
- `test_sft.jsonl`
- `metadata.json`

### 单独评估

```bash
python3 /home/ruoqi/Visual-RFT/classification/eval_galaxy_rft.py \
  --csv_path /home/ruoqi/data/test.csv \
  --model_name_or_path /path/to/checkpoint \
  --processor_name_or_path /path/to/processor_or_sft_model \
  --output_path /tmp/galaxy_predictions.jsonl \
  --num_votes 3 \
  --include_few_shot
```

这会额外输出：

- `*_metrics.json`
- `*_errors.json`
