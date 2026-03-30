# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config


STRUCTURED_FIELDS = [
    "smoothness",
    "shape",
    "spiral_arms",
    "bar",
    "edge_on",
    "bulge",
    "interaction",
]


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "structured_format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'structured_format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


@dataclass
class GRPOClassificationConfig(GRPOConfig):
    max_prompt_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of prompt tokens kept before generation."},
    )


def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass

        if reward == 0.0:
            try:
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                ground_truth = ground_truth.replace(" ", "").replace("_", "").lower()
                student_answer = student_answer.replace(" ", "").replace("_", "").lower()
                if ground_truth in student_answer or student_answer in ground_truth:
                    reward = 1.0
            except Exception:
                pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    return rewards


def structured_format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL)
        answer_match = re.search(r"<answer>.*?</answer>", content, re.DOTALL)
        if think_match is None or answer_match is None:
            rewards.append(0.0)
            continue

        lines = [line.strip() for line in think_match.group(1).splitlines() if line.strip()]
        values = {}
        for line in lines:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

        reward = 1.0 if all(field in values and values[field] for field in STRUCTURED_FIELDS) else 0.0
        rewards.append(reward)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "structured_format": structured_format_reward,
}

SYSTEM_PROMPT = (
    "You are an astronomy assistant for galaxy morphology classification. "
    "Inspect the galaxy image and respond with a structured checklist inside <think> using one key=value pair per line, "
    "followed by exactly one final label inside <answer>."
)


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    from datasets import DatasetDict

    dataset = DatasetDict.load_from_disk(script_args.dataset_name)

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)
        removable_columns = [col for col in ["messages"] if col in dataset[script_args.dataset_train_split].column_names]
        if removable_columns:
            dataset = dataset.remove_columns(removable_columns)
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOClassificationConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
