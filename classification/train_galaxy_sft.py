import argparse
import json
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments


def apply_chat_template_safe(processor, messages, add_generation_prompt: bool):
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


class GalaxySFTDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GalaxySFTCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _assistant_text(self, content: Any) -> str:
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts)
        if isinstance(content, str):
            return content
        return str(content)

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for item in batch:
            image = Image.open(item["image_path"]).convert("RGB")
            prompt_messages = item["messages"][:-1]
            answer_text = self._assistant_text(item["messages"][-1]["content"])
            full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": answer_text}]}]

            prompt_text = apply_chat_template_safe(self.processor, prompt_messages, add_generation_prompt=True)
            full_text = apply_chat_template_safe(self.processor, full_messages, add_generation_prompt=False)

            full_inputs = self.processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = full_inputs["input_ids"][0]
            attention_mask = full_inputs["attention_mask"][0]
            labels = input_ids.clone()
            prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[0])
            labels[:prompt_len] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            pixel_values_list.append(full_inputs["pixel_values"])
            if "image_grid_thw" in full_inputs:
                image_grid_thw_list.append(full_inputs["image_grid_thw"])

        max_len = max(x.shape[0] for x in input_ids_list)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_n = max_len - input_ids.shape[0]
            if pad_n > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_n,), self.pad_token_id, dtype=input_ids.dtype)], dim=0)
                attention_mask = torch.cat([attention_mask, torch.zeros((pad_n,), dtype=attention_mask.dtype)], dim=0)
                labels = torch.cat([labels, torch.full((pad_n,), -100, dtype=labels.dtype)], dim=0)
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

        batch_out = {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "attention_mask": torch.stack(padded_attention_mask, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
        }
        if image_grid_thw_list:
            batch_out["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        return batch_out


def main():
    parser = argparse.ArgumentParser(description="Train a galaxy SFT warmup model for Visual-RFT.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="all-linear")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_implementation,
    )
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = GalaxySFTDataset(args.train_jsonl)
    eval_dataset = GalaxySFTDataset(args.eval_jsonl)
    collator = GalaxySFTCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
