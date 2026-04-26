"""
Quick diagnostic: load model, run 3 SFT steps, report VRAM + step timing.
Usage: python scripts/diagnose_training.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    used = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return {
        "allocated_gb": round(used, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - reserved, 2),
        "utilization_pct": round(used / total * 100, 1),
    }


def main() -> None:
    print("=== Training Diagnostic ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Baseline VRAM: {gpu_stats()}")
    print()

    sys.path.insert(0, str(ROOT / "src"))
    from decomp_clarifier.settings import load_training_config

    config = load_training_config(ROOT, "sft_qwen35_2b")
    print(f"Model:          {config.model.base_model_id}")
    print(f"max_seq_length: {config.training.max_seq_length}")
    print(f"lora_rank:      {config.training.lora_rank}")
    print(f"load_in_4bit:   {config.training.load_in_4bit}")
    print(f"batch_size:     {config.training.batch_size}")
    print(f"grad_accum:     {config.training.grad_accum_steps}")
    print()

    t0 = time.time()
    print("Loading model...")
    import unsloth  # noqa: F401

    from decomp_clarifier.training.sft.model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(config)
    load_time = time.time() - t0
    after_load = gpu_stats()
    print(f"Model loaded in {load_time:.1f}s")
    print(f"VRAM after model load: {after_load}")
    print()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total_params:,} "
        f"({100 * trainable / total_params:.2f}%)"
    )
    print()

    dataset_path = ROOT / "data" / "processed" / "sft" / "sft_records.jsonl"
    if not dataset_path.exists():
        print(f"No dataset at {dataset_path} - skipping training steps")
        return

    print("Loading dataset and running 3 training steps...")
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from decomp_clarifier.training.sft.data import combine_prompt_and_response

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda row: {"text": combine_prompt_and_response(row)})

    class TimingCallback:
        def __init__(self):
            self.step_start = None
            self.steps = []

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self.step_start is None:
                return
            elapsed = time.time() - self.step_start
            vram = gpu_stats()
            self.steps.append(
                {
                    "step": state.global_step,
                    "seconds": round(elapsed, 1),
                    "vram": vram,
                }
            )
            print(
                f"  Step {state.global_step}: {elapsed:.1f}s | "
                f"VRAM: {vram['reserved_gb']:.2f}GB used / "
                f"{vram['total_gb']:.2f}GB"
            )

    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def __init__(self, tc):
            self.tc = tc

        def on_step_begin(self, args, state, control, **kwargs):
            self.tc.on_step_begin(args, state, control, **kwargs)

        def on_step_end(self, args, state, control, **kwargs):
            self.tc.on_step_end(args, state, control, **kwargs)

    tc = TimingCallback()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(ROOT / "artifacts" / "runs" / "diag-sft"),
            max_length=config.training.max_seq_length or 2048,
            per_device_train_batch_size=config.training.batch_size or 2,
            gradient_accumulation_steps=config.training.grad_accum_steps or 4,
            max_steps=3,
            learning_rate=2e-4,
            logging_steps=1,
            report_to="none",
        ),
        dataset_text_field="text",
        callbacks=[_CB(tc)],
    )

    t1 = time.time()
    trainer.train()
    total_train_seconds = time.time() - t1

    print()
    print("=== Results ===")
    for step in tc.steps:
        print(
            f"  Step {step['step']}: {step['seconds']}s - "
            f"VRAM reserved {step['vram']['reserved_gb']:.2f}GB / "
            f"{step['vram']['total_gb']:.2f}GB "
            f"({step['vram']['utilization_pct']}% of total)"
        )
    if tc.steps:
        avg = sum(step["seconds"] for step in tc.steps) / len(tc.steps)
        peak_vram = max(step["vram"]["reserved_gb"] for step in tc.steps)
        print(f"\n  Avg step time:  {avg:.1f}s")
        print(f"  Total train:    {total_train_seconds:.1f}s")
        print(
            f"  Peak VRAM:      {peak_vram:.2f} GB / "
            f"{tc.steps[0]['vram']['total_gb']:.2f} GB"
        )
        print(
            f"  VRAM headroom:  "
            f"{tc.steps[0]['vram']['total_gb'] - peak_vram:.2f} GB free"
        )
        effective_batch = (config.training.batch_size or 2) * (
            config.training.grad_accum_steps or 4
        )
        samples_per_sec = effective_batch / avg
        print(
            f"  Throughput:     {samples_per_sec:.2f} samples/sec "
            f"(effective batch {effective_batch})"
        )


if __name__ == "__main__":
    main()
