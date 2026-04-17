"""
Quick diagnostic: load model, run 3 SFT steps, report VRAM + step timing.
Usage: python scripts/diagnose_training.py
"""
from __future__ import annotations

import json
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

    # Load config
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from decomp_clarifier.settings import load_training_config

    config = load_training_config(ROOT, "sft_qwen35_4b")
    print(f"Model:          {config.model.base_model_id}")
    print(f"max_seq_length: {config.training.max_seq_length}")
    print(f"lora_rank:      {config.training.lora_rank}")
    print(f"load_in_4bit:   {config.training.load_in_4bit}")
    print(f"batch_size:     {config.training.batch_size}")
    print(f"grad_accum:     {config.training.grad_accum_steps}")
    print()

    # Load model
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

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")
    print()

    # Run 3 training steps
    dataset_path = ROOT / "data" / "processed" / "sft" / "sft_records.jsonl"
    if not dataset_path.exists():
        print(f"No dataset at {dataset_path} — skipping training steps")
        return

    print("Loading dataset and running 3 training steps...")
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from decomp_clarifier.training.sft.data import combine_prompt_and_response

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda row: {"text": combine_prompt_and_response(row)})

    step_times = []
    class TimingCallback:
        def __init__(self):
            self.step_start = None
            self.steps = []

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self.step_start is not None:
                elapsed = time.time() - self.step_start
                vram = gpu_stats()
                self.steps.append({"step": state.global_step, "seconds": round(elapsed, 1), "vram": vram})
                print(f"  Step {state.global_step}: {elapsed:.1f}s | VRAM: {vram['reserved_gb']:.2f}GB used / {vram['total_gb']:.2f}GB")

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
    total_train = time.time() - t1

    print()
    print("=== Results ===")
    for s in tc.steps:
        print(f"  Step {s['step']}: {s['seconds']}s — VRAM reserved {s['vram']['reserved_gb']:.2f}GB / {s['vram']['total_gb']:.2f}GB ({s['vram']['utilization_pct']}% of total)")
    if tc.steps:
        avg = sum(s["seconds"] for s in tc.steps) / len(tc.steps)
        peak_vram = max(s["vram"]["reserved_gb"] for s in tc.steps)
        print(f"\n  Avg step time:  {avg:.1f}s")
        print(f"  Peak VRAM:      {peak_vram:.2f} GB / {tc.steps[0]['vram']['total_gb']:.2f} GB")
        print(f"  VRAM headroom:  {tc.steps[0]['vram']['total_gb'] - peak_vram:.2f} GB free")
        samples_per_sec = (config.training.batch_size or 2) * (config.training.grad_accum_steps or 4) / avg
        print(f"  Throughput:     {samples_per_sec:.2f} samples/sec (effective batch {(config.training.batch_size or 2) * (config.training.grad_accum_steps or 4)})")


if __name__ == "__main__":
    main()
