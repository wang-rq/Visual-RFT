#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/ruoqi/Visual-RFT/src/virft/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ATTN_IMPL=sdpa

ROOT_DIR=/home/ruoqi/Visual-RFT
TEST_CSV=/home/ruoqi/data/test.csv
DATASET_DIR=${ROOT_DIR}/share_data/galaxy_rft_smoke_qwen25vl_prepared
INIT_MODEL_PATH=${ROOT_DIR}/share_models/Qwen2.5-VL-7B-Instruct_SFT_galaxy_smoke
PROCESSOR_PATH=${INIT_MODEL_PATH}
GRPO_OUT=${ROOT_DIR}/share_models/Qwen2.5-VL-7B-Instruct_GRPO_galaxy_smoke
RESULT_PATH=${ROOT_DIR}/classification/results/qwen25_vl_galaxy_smoke_predictions.jsonl

RUN_GRPO=${RUN_GRPO:-1}
RUN_EVAL=${RUN_EVAL:-1}

if [ "${RUN_GRPO}" = "1" ]; then
  cd ${ROOT_DIR}/src/virft
  torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12347 \
    -m open_r1.grpo_classification \
    --output_dir ${GRPO_OUT} \
    --model_name_or_path ${INIT_MODEL_PATH} \
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
    --processor_name_or_path ${PROCESSOR_PATH} \
    --output_path ${RESULT_PATH} \
    --max_samples 64 \
    --dtype bfloat16 \
    --attn_implementation ${ATTN_IMPL} \
    --num_votes 3 \
    --include_few_shot \
    --max_samples 500
fi
