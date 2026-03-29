## Galaxy Visual-RFT

端到端 smoke test 流程：

```bash
bash /home/ruoqi/Visual-RFT/src/scripts/qwen3_vl_galaxy_smoke_test.sh
```

这个脚本会完成三步：

1. 从 `/home/ruoqi/data/train.csv` 和 `/home/ruoqi/data/test.csv` 生成适配 Visual-RFT 的 `DatasetDict`
2. 用 `/home/ruoqi/models/Qwen3-VL-8B-Instruct` 在 4 张卡上跑一个小规模 GRPO smoke test
3. 在测试集的 64 个样本上做推理评估，并把结果写到 `classification/results/qwen3_vl_galaxy_smoke_predictions.jsonl`

如果你想单独运行各个步骤：

```bash
python3 /home/ruoqi/Visual-RFT/classification/prepare_galaxy_rft_dataset.py \
  --train_csv /home/ruoqi/data/train.csv \
  --test_csv /home/ruoqi/data/test.csv \
  --output_dir /home/ruoqi/Visual-RFT/share_data/galaxy_rft_custom
```

```bash
cd /home/ruoqi/Visual-RFT/src/virft
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 src/open_r1/grpo_classification.py ...
```

```bash
python3 /home/ruoqi/Visual-RFT/classification/eval_galaxy_rft.py \
  --csv_path /home/ruoqi/data/test.csv \
  --model_name_or_path /path/to/checkpoint \
  --processor_name_or_path /home/ruoqi/models/Qwen3-VL-8B-Instruct \
  --output_path /tmp/galaxy_predictions.jsonl
```
