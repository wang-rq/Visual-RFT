#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/ruoqi/Visual-RFT/src/virft/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ATTN_IMPL=sdpa

ROOT_DIR=/home/ruoqi/Visual-RFT
MODEL_PATH=/home/ruoqi/models/Qwen2.5-VL-7B-Instruct
TRAIN_CSV=/home/ruoqi/data/train.csv
TEST_CSV=/home/ruoqi/data/test.csv
DATASET_DIR=${ROOT_DIR}/share_data/galaxy_rft_smoke_qwen25vl_prepared
SFT_OUT=${ROOT_DIR}/share_models/Qwen2.5-VL-7B-Instruct_SFT_galaxy_smoke
GRPO_OUT=${ROOT_DIR}/share_models/Qwen2.5-VL-7B-Instruct_GRPO_galaxy_smoke
RESULT_PATH=${ROOT_DIR}/classification/results/qwen25_vl_galaxy_smoke_predictions.jsonl

RUN_PREPARE=${RUN_PREPARE:-1}
RUN_SFT=${RUN_SFT:-1}
RUN_GRPO=${RUN_GRPO:-1}
RUN_EVAL=${RUN_EVAL:-1}

/home/ruoqi/anaconda3/envs/qwen_grpo/bin/python - <<'PY'
from pathlib import Path
import sys
from safetensors import safe_open

model_dir = Path("/home/ruoqi/models/Qwen2.5-VL-7B-Instruct")
bad_files = []
for shard in sorted(model_dir.glob("model-*.safetensors")):
    try:
        with safe_open(str(shard), framework="pt"):
            pass
    except Exception as exc:
        bad_files.append(f"{shard.name}: {exc}")

if bad_files:
    print("Model shard validation failed:")
    for item in bad_files:
        print(f"  - {item}")
    sys.exit(1)

print("Model shard validation passed.")
PY

if [ "${RUN_PREPARE}" = "1" ]; then
  python3 ${ROOT_DIR}/classification/prepare_galaxy_rft_dataset.py \
    --train_csv ${TRAIN_CSV} \
    --test_csv ${TEST_CSV} \
    --output_dir ${DATASET_DIR} \
    --max_train_samples 256 \
    --max_test_samples 64 \
    --seed 42
fi

if [ "${RUN_SFT}" = "1" ]; then
  torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    ${ROOT_DIR}/classification/train_galaxy_sft.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_jsonl ${DATASET_DIR}/train_sft.jsonl \
    --eval_jsonl ${DATASET_DIR}/test_sft.jsonl \
    --output_dir ${SFT_OUT} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_steps 40 \
    --logging_steps 10 \
    --eval_steps 20 \
    --save_steps 20 \
    --max_length 2048 \
    --bf16 \
    --attn_implementation ${ATTN_IMPL} \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear
fi

if [ "${RUN_GRPO}" = "1" ]; then
  cd ${ROOT_DIR}/src/virft
  torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12347 \
    -m open_r1.grpo_classification \
    --output_dir ${GRPO_OUT} \
    --model_name_or_path ${SFT_OUT} \
    --dataset_name ${DATASET_DIR} \
    --dataset_train_split train \
    --dataset_test_split test \
    --deepspeed local_scripts/zero2_torch_adam.json \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation ${ATTN_IMPL} \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --run_name Qwen2.5-VL-7B_GRPO_galaxy_smoke \
    --save_steps 20 \
    --save_only_model true \
    --eval_strategy no \
    --num_generations 8 \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear
  cd ${ROOT_DIR}
fi

if [ "${RUN_EVAL}" = "1" ]; then
  python3 ${ROOT_DIR}/classification/eval_galaxy_rft.py \
    --csv_path ${TEST_CSV} \
    --model_name_or_path ${GRPO_OUT} \
    --processor_name_or_path ${SFT_OUT} \
    --output_path ${RESULT_PATH} \
    --max_samples 64 \
    --dtype bfloat16 \
    --attn_implementation ${ATTN_IMPL} \
    --num_votes 3 \
    --include_few_shot
fi
