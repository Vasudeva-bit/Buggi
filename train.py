import glob
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from buggi import DOT, DOTConfig

out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2"
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 8
block_size = 1024
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
device = "cuda"
dtype = "float16"
compile = False
data_dir = "data_shards"

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


def get_next_model_number():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    existing_numbers = []
    for filename in os.listdir(out_dir):
        if filename.endswith(".png") or filename.endswith(".txt"):
            try:
                number = int(filename.split(".")[0])
                existing_numbers.append(number)
            except ValueError:
                continue

    return max(existing_numbers, default=0) + 1


def save_loss_curve_and_details(
    model_number, loss_history, step_history, model_config, training_details
):
    plt.figure(figsize=(12, 8))
    plt.plot(step_history, loss_history, "b-", linewidth=1.5, alpha=0.8)
    plt.title(f"Training Loss - Model {model_number}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    final_loss = loss_history[-1] if loss_history else 0
    best_loss = min(loss_history) if loss_history else 0
    avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0

    stats_text = f"Final: {final_loss:.4f}\nBest: {best_loss:.4f}\nAvg: {avg_loss:.4f}\nSteps: {len(step_history)}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plot_path = os.path.join(out_dir, f"{model_number}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    details_path = os.path.join(out_dir, f"{model_number}.txt")
    with open(details_path, "w") as f:
        f.write(f"Model {model_number}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Model Config:\n")
        for key, value in model_config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")

        f.write("Training Config:\n")
        for key, value in training_details.items():
            f.write(f"{key}: {value}\n")

        f.write(f"\nLoss Stats:\n")
        f.write(f"Final: {final_loss:.6f}\n")
        f.write(f"Best: {best_loss:.6f}\n")
        f.write(f"Average: {avg_loss:.6f}\n")
        f.write(f"Steps: {len(step_history)}\n")

    return plot_path, details_path


class Data(Dataset):
    def __init__(self, data_dir, block_size, shard_index=0):
        self.block_size = block_size
        shard_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not shard_files:
            raise Exception(f"No .bin files found in {data_dir}")

        if shard_index >= len(shard_files):
            raise Exception(f"Shard index {shard_index} not found")

        shard_file = shard_files[shard_index]
        self.data = np.fromfile(shard_file, dtype=np.uint16)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)
        )
        return x, y


def create_shard_loader(shard_index):
    shard_dataset = Data(data_dir, block_size, shard_index)
    shard_loader = DataLoader(
        shard_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return shard_dataset, shard_loader


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


current_model_number = get_next_model_number()

training_config = {
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "weight_decay": weight_decay,
    "beta1": beta1,
    "beta2": beta2,
    "grad_clip": grad_clip,
    "warmup_iters": warmup_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,
    "eval_interval": eval_interval,
    "dtype": dtype,
    "compile": compile,
}

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == "scratch":
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 16768
    gptconf = DOTConfig(**model_args)
    model = DOT(gptconf)
elif init_from == "resume":
    ckpt_path = os.path.join(out_dir, "full_checkpoint.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = DOTConfig(**model_args)
    model = DOT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

model.to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if compile:
    unoptimized_model = model
    model = torch.compile(model)


def train_sequential_shards(model, optimizer, device, max_iters_total, model_number):
    model.to(device)
    shard_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    num_shards = len(shard_files)
    max_iters_per_shard = max_iters_total // num_shards

    global_step = 0
    total_loss = 0
    best_loss = float("inf")
    loss_history = []
    step_history = []

    for shard_idx in range(num_shards):
        shard_dataset, shard_loader = create_shard_loader(shard_idx)
        model.train()
        shard_step = 0
        shard_loss = 0

        if device == "cuda":
            torch.cuda.empty_cache()

        for epoch in range(999):
            for batch_idx, (x, y) in enumerate(shard_loader):
                if shard_step >= max_iters_per_shard:
                    break

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                with ctx:
                    logits, loss = model(x, targets=y)

                scaler.scale(loss).backward()

                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                shard_loss += loss.item()
                global_step += 1
                shard_step += 1

                loss_history.append(loss.item())
                step_history.append(global_step)

                if global_step % log_interval == 0:
                    avg_loss = total_loss / global_step
                    shard_avg = shard_loss / shard_step
                    print(
                        f"Model {model_number} | Shard {shard_idx} | Step {shard_step:4d} | Global {global_step:6d} | LR: {lr:.2e} | Loss: {loss.item():.4f} | Avg: {shard_avg:.4f}"
                    )

                if global_step % eval_interval == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "current_shard": shard_idx,
                        "shard_step": shard_step,
                        "loss": loss.item(),
                        "best_loss": best_loss,
                        "config": config,
                        "model_args": model_args,
                        "model_number": model_number,
                        "loss_history": loss_history,
                        "step_history": step_history,
                    }

                    full_checkpoint_path = os.path.join(out_dir, "full_checkpoint.pth")
                    torch.save(checkpoint, full_checkpoint_path)

                    if device == "cuda":
                        torch.cuda.empty_cache()

            if shard_step >= max_iters_per_shard:
                break

    final_checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "completed_shards": num_shards,
        "total_shards": num_shards,
        "final_loss": total_loss / global_step if global_step > 0 else 0,
        "best_loss": best_loss,
        "config": config,
        "model_args": model_args,
        "model_number": model_number,
        "loss_history": loss_history,
        "step_history": step_history,
    }

    final_path = os.path.join(out_dir, "full_checkpoint.pth")
    torch.save(final_checkpoint, final_path)

    plot_path, details_path = save_loss_curve_and_details(
        model_number, loss_history, step_history, model_args, training_config
    )

    return total_loss / global_step if global_step > 0 else 0


if device == "cuda":
    torch.cuda.empty_cache()

train_sequential_shards(model, optimizer, device, max_iters, current_model_number)
